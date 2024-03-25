import os
import os.path as osp

from collections import OrderedDict
import logging

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

Loader, Dumper = OrderedYaml()

def simple_parse(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    return opt

def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['num_gpu'] = len(opt['gpu_ids'])
    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        opt['network_G']['scale'] = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        ###
        dataset['batch_size'] = dataset.get('batch_size_per_gpu',1) * len(opt['gpu_ids'])
        ###

        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if opt['distortion'] == 'sr':
            dataset['scale'] = opt['scale']

        is_lmdb = False
        if dataset.get('dataroot_GT') is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ') is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'disk'

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['debug'] = True
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
            # torch.autograd.set_detect_anomaly(True)
        else:
            opt['debug'] = False
    else:  # test
        results_root = osp.join(opt['path']['root'], 'test_results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    return opt

def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        # if 'gan' in opt['model']:
        #     opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
        #                                                '{}_D.pth'.format(resume_iter))
        #     logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])


def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

def save_network(path, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(path, save_filename)
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

def save_training_state(path, epoch, iter_step, optimizer, scheduler, best_record):
    '''Saves training state during training, which will be used for resuming'''
    state = {'epoch'      : epoch,
             'iter'       : iter_step,
             'scheduler'  : scheduler.state_dict(),
             'optimizer'  : optimizer.state_dict(),
             'best_record': best_record}
    save_filename = '{}.state'.format(iter_step)
    save_path = os.path.join(path, save_filename)
    torch.save(state, save_path)
from easydict import EasyDict as edict
import importlib
import os

from utils.common import scandir


dataset_root = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [
        os.path.splitext(os.path.basename(v))[0] for v in scandir(dataset_root)
        if v.endswith('_dataset.py')
]
_dataset_modules = [
        importlib.import_module(f'dataset.{file_name}')
        for file_name in dataset_filenames
]


class DATASET:
    LEGAL = ['DIV2K', 'Flickr2K', 'Set5', 'Set14', 'BSDS100', 'Urban100', 'Manga109', 'BSDS1', 'CSDS1']

    # training dataset
    DIV2K = edict()
    DIV2K.TRAIN = edict()
    DIV2K.TRAIN.HRx2 = '/home/ubuntu/Downloads/SRdataset/DIV2K/DIV2K_train_HR_sub'  # 32208
    DIV2K.TRAIN.HRx3 = '/home/ubuntu/Downloads/SRdataset/DIV2K/DIV2K_train_HR_sub'  # 32208
    DIV2K.TRAIN.HRx4 = '/home/ubuntu/Downloads/SRdataset/DIV2K/DIV2K_train_HR_sub'  # 32208
    DIV2K.TRAIN.LRx2 = '/home/ubuntu/Downloads/SRdataset/DIV2K/DIV2K_train_LR_bicubic_sub/X2'
    DIV2K.TRAIN.LRx3 = '/home/ubuntu/Downloads/SRdataset/DIV2K/DIV2K_train_LR_bicubic_sub/X3'
    DIV2K.TRAIN.LRx4 = '/home/ubuntu/Downloads/SRdataset/DIV2K/DIV2K_train_LR_bicubic_sub/X3'

    Flickr2K = edict()
    Flickr2K.TRAIN = edict()
    Flickr2K.TRAIN.HRx2 = '/home/ubuntu/Downloads/SRdataset/Flickr2K/Flickr2K_HR_sub'  # 106641
    Flickr2K.TRAIN.HRx3 = '/home/ubuntu/Downloads/SRdataset/Flickr2K/Flickr2K_HR_sub'  # 106641
    Flickr2K.TRAIN.HRx4 = '/home/ubuntu/Downloads/SRdataset/Flickr2K/Flickr2K_HR_sub'  # 106641
    Flickr2K.TRAIN.LRx2 = '/home/ubuntu/Downloads/SRdataset/Flickr2K/Flickr2K_LR_bicubic_sub/X2'
    Flickr2K.TRAIN.LRx3 = '/home/ubuntu/Downloads/SRdataset/Flickr2K/Flickr2K_LR_bicubic_sub/X3'
    Flickr2K.TRAIN.LRx4 = '/home/ubuntu/Downloads/SRdataset/Flickr2K/Flickr2K_LR_bicubic_sub/X4'

    # testing dataset
    Set5 = edict()
    Set5.VAL = edict()
    Set5.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set5\\HR\\mod8'
    Set5.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set5\\HR\\mod12'
    Set5.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set5\\HR\\mod16'
    Set5.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set5\\LR_bicubic\\X2'
    Set5.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set5\\LR_bicubic\\X3'
    Set5.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set5\\LR_bicubic\\X4'

    Set14 = edict()
    Set14.VAL = edict()
    Set14.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set14\\HR\\mod8'
    Set14.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set14\\HR\\mod12'
    Set14.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set14\\HR\\mod16'
    Set14.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set14\\LR_bicubic\\X2'
    Set14.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set14\\LR_bicubic\\X3'
    Set14.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Set14\\LR_bicubic\\X4'

    BSDS100 = edict()
    BSDS100.VAL = edict()
    BSDS100.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\B100\\HR\\mod8'
    BSDS100.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\B100\\HR\\mod12'
    BSDS100.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\B100\\HR\\mod16'
    BSDS100.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\B100\\LR_bicubic\\X2'
    BSDS100.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\B100\\LR_bicubic\\X3'
    BSDS100.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\B100\\LR_bicubic\\X4'

    BSDS1 = edict()
    BSDS1.VAL = edict()
    BSDS1.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\B1\\HR\\mod2'
    BSDS1.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\B1\\HR\\mod3'
    BSDS1.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\B1\\HR\\mod4'
    BSDS1.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\B1\\LR_bicubic\\X2'
    BSDS1.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\B1\\LR_bicubic\\X3'
    BSDS1.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\B1\\LR_bicubic\\X4'

    CSDS1 = edict()
    CSDS1.VAL = edict()
    CSDS1.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\C1\\HR\\mod2'
    CSDS1.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\C1\\HR\\mod3'
    CSDS1.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\C1\\HR\\mod4'
    CSDS1.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\C1\\LR_bicubic\\X2'
    CSDS1.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\C1\\LR_bicubic\\X3'
    CSDS1.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\C1\\LR_bicubic\\X4'

    Urban100 = edict()
    Urban100.VAL = edict()
    Urban100.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Urban100\\HR\\mod8'
    Urban100.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Urban100\\HR\\mod12'
    Urban100.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Urban100\\HR\\mod16'
    Urban100.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Urban100\\LR_bicubic\\X2'
    Urban100.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Urban100\\LR_bicubic\\X3'
    Urban100.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Urban100\\LR_bicubic\\X4'

    Manga109 = edict()
    Manga109.VAL = dict()
    Manga109.VAL.HRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Manga109\\HR\\mod8'
    Manga109.VAL.HRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Manga109\\HR\\mod12'
    Manga109.VAL.HRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Manga109\\HR\\mod16'
    Manga109.VAL.LRx2 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Manga109\\LR_bicubic\\X2'
    Manga109.VAL.LRx3 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Manga109\\LR_bicubic\\X3'
    Manga109.VAL.LRx4 = 'D:\\py project\\Simple-SR-master\\benchmark_SR\\Manga109\\LR_bicubic\\X4'


def get_dataset(config):
    dataset_type = config.TYPE
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    hr_paths = []
    lr_paths = []
    D = DATASET()

    for dataset, split in zip(config.DATASETS, config.SPLITS):
        if dataset not in D.LEGAL or split not in eval('D.%s' % dataset):
            raise ValueError('Illegal dataset.')
        hr_paths.append(eval('D.%s.%s.HRx%d' % (dataset, split, config.SCALE)))
        lr_paths.append(eval('D.%s.%s.LRx%d' % (dataset, split, config.SCALE)))

    return dataset_cls(hr_paths, lr_paths, config)

def get_dataset_1(config,x,y):
    dataset_type = config.TYPE
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    hr_paths = []
    lr_paths = []
    D = DATASET()

    dataset, split = x, y
    if dataset not in D.LEGAL or split not in eval('D.%s' % dataset):
        raise ValueError('Illegal dataset.')
    hr_paths.append(eval('D.%s.%s.HRx%d' % (dataset, split, config.SCALE)))
    lr_paths.append(eval('D.%s.%s.LRx%d' % (dataset, split, config.SCALE)))

    return dataset_cls(hr_paths, lr_paths, config)


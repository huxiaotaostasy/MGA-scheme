import os
import argparse
import logging
from importlib import import_module
import torch

from utils import util, options, modelsummary
from exps.rfdn.RFDNMGA import RFDNBase
import time


'''
This code can help you to calculate:
`FLOPs`, `#Params`, `Runtime`, `#Activations`, `#Conv`, and `Max Memory Allocated`.

- `#Params' denotes the total number of parameters. 
- `FLOPs' is the abbreviation for floating point operations. 
- `#Activations' measures the number of elements of all outputs of convolutional layers. 
- `Memory' represents maximum GPU memory consumption according to the PyTorch function torch.cuda.max_memory_allocated().
- `#Conv' represents the number of convolutional layers. 
- `FLOPs', `#Activations', and `Memory' are tested on an LR image of size 256x256.

For more information, please refer to ECCVW paper "AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results".

# If you use this code, please consider the following citations:

@inproceedings{zhang2020aim,
  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Kai Zhang and Martin Danelljan and Yawei Li and Radu Timofte and others},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
@inproceedings{zhang2019aim,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}

CuDNN (https://developer.nvidia.com/rdp/cudnn-archive) should be installed.

For `Memery` and `Runtime`, set 'print_modelsummary = False' and 'save_results = False'.
'''


def get_image_file_paths(dir, extension):
    assert os.path.isdir(dir)
    img_paths = []
    for dirpath, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if fname.endswith(extension):
                img_path = os.path.join(dirpath, fname)
                img_paths.append(img_path)
    assert img_paths, '{:s} has no valid image file'.format(dir)
    return sorted(img_paths)

def main():

    # --------------------------------
    # basic settings
    # --------------------------------

    # logger
    util.setup_logger('stat', root=None, phase='summary', level=logging.INFO, screen=True, tofile=False)
    logger = logging.getLogger('stat')

    logger.info('{:>16s} : '.format('Model configurations'))

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)      # set GPU ID
        logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        logger.info('CUDA is NOT available, run on cpu.')
        device = torch.device('cpu')

    # --------------------------------
    # define network and load model
    # --------------------------------

    model = RFDNBase()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # --------------------------------
    # print model summary
    # --------------------------------
    logger.info('{:>16s} : '.format('Model Structure'))
    logger.info(str(model))

    input_dim = (3, 1024, 512)

    activations, num_conv2d = modelsummary.get_model_activation(model, input_dim)
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
    logger.info('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

    flops = modelsummary.get_model_flops(model, input_dim, False)
    logger.info('{:>16s} : {:<.4f} [G]'.format('#FLOPs', flops/10**9))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))


if __name__ == '__main__':

    main()

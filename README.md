# Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks
## Introduction
This project is the implement of [Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790072.pdf).

[ECCV Poster](https://drive.google.com/file/d/1lYuRgh2v-k_7eWNiJ5OaUsEj813jQB-r/view?usp=share_link) | [ECCV 5-min presentation](https://youtu.be/ICHXL6BUYGI)

## Requirements
### Train Data
- DIV2K
- Flickr2K
Please download the datasets and put them in the `data` folder.
Please refer to the [official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for DIV2K and [official website](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) for Flickr2K.
Please note that changing the path of the datasets in the code (`__init__.py` in `dataset` folder) is necessary.
### Test Data
Download Set5, Set14, Urban100, BSDS100 and Manga109 from [Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) uploaded by BasicSR.
Update the dataset location in .dataset/__init__.py.
## Training
### Train the model
To train the model, run the following command:
```
python3 -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 train_all.py 
python3 -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 train_mask.py 
```
## Testing
Please refer to validate.py in each experiment folder or quick test above.

## FLOPs and Parameters
Please run the following command to get the FLOPs and Parameters of the model:
```
python3 cal_flops_params.py
```
For more information, please refer to ECCVW paper "AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results".
CuDNN (https://developer.nvidia.com/rdp/cudnn-archive) should be installed.

## Acknowledgement
We refer to [BasicSR](https://github.com/xinntao/BasicSR) and [Simple-SR](https://github.com/dvlab-research/Simple-SR) for some details.
Thanks for [Kai Zhang](https://cszn.github.io/) for providing the code of calculating FLOPs and Parameters.

## Citation

```
@inproceedings{hu2022restore,
  title={Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks},
  author={Hu, Xiaotao and Xu, Jun and Gu, Shuhang and Cheng, Ming-Ming and Liu, Li},
  booktitle={European Conference on Computer Vision},
  pages={74--91},
  year={2022},
  organization={Springer}
}
```
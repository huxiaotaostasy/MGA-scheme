import os
import cv2
import glob
import numpy as np
def mod_crop(img_in, scale):
    img=np.copy(img_in)
    if img.ndim==2:
        H,W=img.shape
        H_r, W_r=H%scale, W%scale
        img=img[:H-H_r,:W-W_r]
    elif img.ndim==3:
        H,W,C=img.shape
        H_r, W_r=H%scale, W%scale
        img=img[:H-H_r,:W-W_r,:]
    return img

if __name__=='__main__':
    mod_num = 'mod12'
    dataset_path = '/home/ubuntu/Downloads/SRdataset/benchmark_SR/B100/HR/or/'
    output_path = os.path.join('/home/ubuntu/Downloads/SRdataset/benchmark_SR/B100/HR', mod_num)
    os.mkdir(output_path)
    l=glob.glob(os.path.join(dataset_path,'*.png'))
    for name in l:
        img_name=name.split('/')[-1]
        img1=cv2.imread(name)
        img=mod_crop(img1,eval(mod_num[-1]))
        cv2.imwrite(os.path.join(output_path,img_name),img)


import cv2
import os
import numpy as np
from utils import util, options, modelsummary
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')
import time
import torch
import torch.nn.functional as F

from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr

def compute_loss(coarse_img, out_err, hr_img):
    true_err = torch.abs(coarse_img.sum(dim=1).unsqueeze(dim=1).detach() - hr_img.sum(dim=1).unsqueeze(dim=1))
    return dict(err_loss=F.mse_loss(out_err, true_err),chan_loss=F.l1_loss(coarse_img, hr_img))

def validate_results(model, val_loader, device, save_path='.'):
    with torch.no_grad():
        psnr_l = []
        ssim_l = []
        for idx, (lr_img, hr_img) in enumerate(val_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            model.eval()
            output = model(lr_img)
            output = tensor2img(output)
            gt = tensor2img(hr_img)

            ipath = os.path.join(save_path, '%d.png' % (idx))
            cv2.imwrite(ipath, output)

            output = output.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0
            output = bgr2ycbcr(output, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            psnr = calculate_psnr(output * 255, gt * 255)
            ssim = calculate_ssim(output * 255, gt * 255)
            psnr_l.append(psnr)
            ssim_l.append(ssim)
    return psnr_l, ssim_l

def validate_2(model, val_loader, config, device, iteration, save_path='.'):
    with torch.no_grad():
        psnr_l = []
        ssim_l = []
        psnr_l_2 = []
        ssim_l_2 = []
        flops_l = []
        times_l = []

        for idx, (lr_img, hr_img) in enumerate(val_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            model.eval()
            s=time.time()

            refine_img, coarse_img = model(lr_img, is_return_coarse=True)

            t=time.time()

            for k, v in model.named_parameters():
                v.requires_grad = False
            model = model.to(device)
            b,c,h,w=lr_img.shape
            input_dim=(c,h,w)
            flops = modelsummary.get_model_flops(model, input_dim, False)

            output = tensor2img(refine_img)
            output2 = tensor2img(coarse_img)
            gt = tensor2img(hr_img)

            if config.VAL.SAVE_IMG:
                ipath = os.path.join(save_path, '%d_%03d.png' % (iteration, idx))
                cv2.imwrite(ipath, np.concatenate([output, gt], axis=1))

            output = output.astype(np.float32) / 255.0
            output2 = output2.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            if config.VAL.TO_Y:
                output = bgr2ycbcr(output, only_y=True)
                output2 = bgr2ycbcr(output2, only_y=True)
                gt = bgr2ycbcr(gt, only_y=True)

            if config.VAL.CROP_BORDER != 0:
                cb = config.VAL.CROP_BORDER
                output = output[cb:-cb, cb:-cb]
                output2 = output2[cb:-cb, cb:-cb]
                gt = gt[cb:-cb, cb:-cb]

            psnr = calculate_psnr(output * 255, gt * 255)
            psnr2 = calculate_psnr(output2 * 255, gt * 255)
            ssim = calculate_ssim(output * 255, gt * 255)
            ssim2 = calculate_ssim(output2 * 255, gt * 255)
            psnr_l.append(psnr)
            psnr_l_2.append(psnr2)
            ssim_l.append(ssim)
            ssim_l_2.append(ssim2)
            flops_l.append(flops)
            times_l.append(t-s)

        avg_psnr = sum(psnr_l) / len(psnr_l)
        avg_ssim = sum(ssim_l) / len(ssim_l)
        avg_psnr_2 = sum(psnr_l_2) / len(psnr_l_2)
        avg_ssim_2 = sum(ssim_l_2) / len(ssim_l_2)
        avg_flops = sum(flops_l) / len(flops_l)
        avg_time = sum(times_l) / len(times_l)

    return avg_psnr, avg_ssim, avg_psnr_2, avg_ssim_2, avg_flops, avg_time

def validate_mask(model, val_loader, config, device, iteration, save_path='.'):
    with torch.no_grad():
        psnr_l = []
        ssim_l = []
        psnr_l_2 = []
        ssim_l_2 = []
        flops_l = []
        times_l = []

        for idx, (lr_img, hr_img) in enumerate(val_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            model.eval()
            s=time.time()

            refine_img_or, refine_img, coarse_img, mask = model(lr_img, is_return_coarse=True)

            t=time.time()

            for k, v in model.named_parameters():
                v.requires_grad = False
            model = model.to(device)
            b,c,h,w=lr_img.shape
            input_dim=(c,h,w)

            flops = modelsummary.get_model_flops(model, input_dim, False)


            output = tensor2img(refine_img)
            output2 = tensor2img(coarse_img)
            gt = tensor2img(hr_img)

            if config.VAL.SAVE_IMG:
                ipath = os.path.join(save_path, '%d_%03d.png' % (iteration, idx))
                cv2.imwrite(ipath, np.concatenate([output, gt], axis=1))

            output = output.astype(np.float32) / 255.0
            output2 = output2.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            if config.VAL.TO_Y:
                output = bgr2ycbcr(output, only_y=True)
                output2 = bgr2ycbcr(output2, only_y=True)
                gt = bgr2ycbcr(gt, only_y=True)

            if config.VAL.CROP_BORDER != 0:
                cb = config.VAL.CROP_BORDER
                output = output[cb:-cb, cb:-cb]
                output2 = output2[cb:-cb, cb:-cb]
                gt = gt[cb:-cb, cb:-cb]

            psnr = calculate_psnr(output * 255, gt * 255)
            psnr2 = calculate_psnr(output2 * 255, gt * 255)
            ssim = calculate_ssim(output * 255, gt * 255)
            ssim2 = calculate_ssim(output2 * 255, gt * 255)
            psnr_l.append(psnr)
            psnr_l_2.append(psnr2)
            ssim_l.append(ssim)
            ssim_l_2.append(ssim2)
            flops_l.append(flops)
            times_l.append(t-s)

        avg_psnr = sum(psnr_l) / len(psnr_l)
        avg_ssim = sum(ssim_l) / len(ssim_l)
        avg_psnr_2 = sum(psnr_l_2) / len(psnr_l_2)
        avg_ssim_2 = sum(ssim_l_2) / len(ssim_l_2)
        avg_flops = sum(flops_l) / len(flops_l)
        avg_time = sum(times_l) / len(times_l)

    return avg_psnr, avg_ssim, avg_psnr_2, avg_ssim_2, avg_flops, avg_time

if __name__ == '__main__':
    from config import config
    from network import Network
    from dataset import get_dataset
    from utils import dataloader
    from utils.model_opr import load_model

    config.VAL.DATASET = 'Set5'

    model = Network(config)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    model_path = 'log/models/200000.pth'
    load_model(model, model_path, cpu=True)
    sys.exit()

    val_dataset = get_dataset(config.VAL)
    val_loader = dataloader.val_loader(val_dataset, config, 0, 1)
    psnr, ssim = validate(model, val_loader, config, device, 0, save_path='.')
    print('PSNR: %.4f, SSIM: %.4f' % (psnr, ssim))

from torch import nn
import torch
from torch.nn import functional as F

import exps.rfdn.block as B
from exps.MPModule import MPModule

class generate_mask_all_patches(nn.Module):
    def __init__(self, padding=1, size=4):
        '''
        nf:num_feature, k:num_point, k1:adaptive_kernel, size:patch_size
        '''
        super(generate_mask_all_patches, self).__init__()
        self.padding=padding
        self.size=size
    def crop_patch(self, feature, idx):
        '''
        feature:B C Hs Ws
        idx:P P P(Batch,Hs,Ws)
        output:
        patches:P C size size
        '''
        feature = F.pad(feature, pad=(self.padding, self.padding, self.padding, self.padding))
        feature = feature.permute(0, 2, 3, 1).unfold(1, self.size+2*self.padding, self.size).unfold(2, self.size+2*self.padding, self.size).contiguous()  # B Hs Ws C size size
        return feature[idx[0], idx[1], idx[2]].contiguous()
    def forward(self, out_lr):
        '''
        out_lr:B C H W
        patches:K C size size
        '''
        B, C, H, W = out_lr.shape
        final_mask = torch.ones(B,1,H//self.size,W//self.size)
        ref = final_mask.squeeze(dim=1)
        idx = torch.nonzero(ref, as_tuple=False)
        idx = idx[:, 0], idx[:, 1], idx[:, 2]  # P Hs Ws
        patches = self.crop_patch(out_lr, idx)  # P C self.size+2*self.padding self.size+2*self.padding
        return patches, idx
    
class RFDNBase(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=4, out_nc=3, upscale=4, return_mid=False):
        super(RFDNBase, self).__init__()
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.return_mid = return_mid
    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output, out_lr

class RFDNRefine(nn.Module):
    def __init__(self, nf=50, scale=4):
        super(RFDNRefine, self).__init__()
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * 2, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 0)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, 3, upscale_factor=scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def replace_patch(self, coarse_img, patches, idx):
        oB,oC,oH,oW=coarse_img.shape
        pB,pC,pH,pW=patches.shape
        coarse_img=coarse_img.view(oB,oC,oH//pH,pH,oW//pW,pW).permute(0,2,4,1,3,5)
        coarse_img[idx[0],idx[1],idx[2]]=patches
        coarse_img=coarse_img.permute(0,3,1,4,2,5).view(oB,oC,oH,oW)
        return coarse_img

    def forward(self, coarse_img, patches, idx):
        coarse_img_1 = coarse_img.clone()
        out_B1 = self.B1(patches)
        out_B2 = self.B2(out_B1)
        out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
        patches = self.LR_conv(out_B)+patches
        patches = self.conv2(self.lrelu(self.conv1(patches)))
        patches = self.upsampler(patches)
        refine_img = self.replace_patch(coarse_img_1, patches, idx)
        return refine_img

class RFDNMGA(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4, padding=1, size=4):
        super(RFDNMGA, self).__init__()
        self.base = RFDNBase(in_nc, nf, num_modules, out_nc, upscale, return_mid=True)
        self.mask = MPModule(nf, padding=padding, size=size)
        self.refine = RFDNRefine(nf, upscale)
    def forward(self,input):
        coarse_img, out_lr = self.base(input)
        patches, idx, mask = self.mask(out_lr)
        refine_img = self.refine(coarse_img, patches, idx)
        return refine_img
    
class RFDNAll(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4, padding=1, size=4):
        super(RFDNAll, self).__init__()
        self.base = RFDNBase(in_nc, nf, num_modules, out_nc, upscale, return_mid=True)
        self.mask = generate_mask_all_patches(padding=padding, size=size)
        self.refine = RFDNRefine(nf, upscale)
    def forward(self,input):
        coarse_img, out_lr = self.base(input)
        patches, idx, mask = self.mask(out_lr)
        refine_img = self.refine(coarse_img, patches, idx)
        return refine_img
if __name__ == '__main__':
    x = torch.randn((2, 3, 64, 64))
    net = RFDNMGA()
    sr = net(x)
    print(sr.shape)
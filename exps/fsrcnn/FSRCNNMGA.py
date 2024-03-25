import math
from torch import nn
import torch
from torch.nn import functional as F
from exps.MPModule import MPModule

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding,
                                       stride=stride).view(n_x, -1, h_out * w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1)

    out = adder.apply(W_col, X_col)

    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col, X_col)
        output = -(W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col / grad_W_col.norm(p=2).clamp(min=1e-12) * math.sqrt(W_col.size(1) * W_col.size(0)) / 5
        grad_X_col = (-(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)

        return grad_W_col, grad_X_col


class adder2d(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output

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
    
class FSRCNNBase(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, m=2):
        super(FSRCNNBase, self).__init__()
        self.first_part = nn.Sequential(
            adder2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [adder2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([adder2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        # self.mid_part.extend([adder2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = [adder2d(s, d, kernel_size=1), nn.PReLU(d)]
        self.last_part.extend([nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)])
        self.last_part = nn.Sequential(*self.last_part)

    def forward(self, x):
        x = self.first_part(x)
        mid = self.mid_part(x)
        output = self.last_part(mid)
        return output


class FSRCNNRefine(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, m=2):
        super(FSRCNNRefine, self).__init__()
        self.mid_part = []
        for _ in range(m):
            self.mid_part.extend([adder2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = [adder2d(s, d, kernel_size=1), nn.PReLU(d)]
        self.last_part.extend([nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)])
        self.last_part = nn.Sequential(*self.last_part)

        self.conv1 = adder2d(s, s, 3, 1, 1)
        self.conv2 = adder2d(s, s, 3, 1, 0)
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
        res = self.mid_part(patches)
        res = self.conv2(self.lrelu(self.conv1(res)))
        patches = self.last_part(res)
        refine_img = self.replace_patch(coarse_img_1, patches, idx)
        return refine_img


class FSRCNNMGA(nn.Module):
    '''
    n_resgroups=7, n_resblocks=20, n_feats=64, reduction=16, scale=4,
                n_colors=3, res_scale=1, conv=common.default_conv, return_mid=True
    '''
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, nb_1=3, nb_2=1, padding=1, size=4):
        super(FSRCNNMGA, self).__init__()
        self.base = FSRCNNBase(scale_factor, num_channels=num_channels, d=d, s=s, m=nb_1, return_mid=True)
        self.mask = MPModule(s, padding=padding, size=size)
        self.refine = FSRCNNRefine(scale_factor, num_channels=num_channels, d=d, s=s, m=nb_2)
    def forward(self,input):
        coarse_img, out_lr = self.base(input)
        patches, idx, mask = self.mask(out_lr)
        refine_img = self.refine(coarse_img, patches, idx)
        return refine_img

class FSRCNNAll(nn.Module):
    '''
    n_resgroups=7, n_resblocks=20, n_feats=64, reduction=16, scale=4,
                n_colors=3, res_scale=1, conv=common.default_conv, return_mid=True
    '''
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, nb_1=3, nb_2=1, padding=1, size=4):
        super(FSRCNNMGA, self).__init__()
        self.base = FSRCNNBase(scale_factor, num_channels=num_channels, d=d, s=s, m=nb_1, return_mid=True)
        self.mask = generate_mask_all_patches(padding=padding, size=size)
        self.refine = FSRCNNRefine(scale_factor, num_channels=num_channels, d=d, s=s, m=nb_2)
    def forward(self,input):
        coarse_img, out_lr = self.base(input)
        patches, idx, mask = self.mask(out_lr)
        refine_img = self.refine(coarse_img, patches, idx)
        return refine_img

# if __name__=="__main__":
#     x=torch.randn(1,3,64,64)
#     net=Model12_mask_only_test_time(scale_factor=4)
#     # print(net)
#     refine_img = net(x)
#     print(refine_img.shape)


from torch import nn
import torch
from torch.nn import functional as F

class MPModule(nn.Module):
    def __init__(self, nf=50, padding=1, size=4):
        super(MPModule, self).__init__()
        self.k1 = 5
        self.spa_mask_1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(True), nn.Conv2d(nf, 2, 3, 1, 1))
        self.conv_kernel = nn.Conv2d(1, self.k1 ** 2, 1, 1, 0)
        self.padding = padding
        self.size = size
        self.t = 1
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def softmax_t(self, x, dim, t):
        x = x / t
        x = x.softmax(dim)
        return x

    def _update_t(self, t):
        self.t = t

    def select_refinement_regions(self, mask: torch.Tensor):
        b, c, h, w = mask.shape
        err = mask.view(b, -1)
        idx = err.topk(240, dim=1, sorted=False).indices
        # idx = err.topk(1000, dim=1, sorted=False).indices #240 for BSDS100 Set5 Set14, 1000 for Urban100 Manga109
        ref = torch.zeros_like(err)
        ref.scatter_(1, idx, 1.)
        ref = ref.view(b, 1, h, w)
        return ref

    def crop_patch(self, feature, idx):
        feature = F.pad(feature, pad=(self.padding, self.padding, self.padding, self.padding))
        feature = feature.permute(0, 2, 3, 1).unfold(1, self.size+2*self.padding, self.size).unfold(2, self.size+2*self.padding, self.size).contiguous()  # B Hs Ws C size size
        return feature[idx[0], idx[1], idx[2]].contiguous()

    def forward(self, out_lr):
        B, C, H, W = out_lr.shape
        spa_mask = self.spa_mask_1(out_lr)
        spa_mask = self.softmax_t(spa_mask, 1, self.t)[:, 1:, ...]
        spa_mask = F.adaptive_avg_pool2d(spa_mask, (H // self.size, W // self.size))
        spa_mask = self.lrelu(spa_mask)

        spa_kernel = self.conv_kernel(spa_mask)
        spa_kernel = spa_kernel.unsqueeze(dim=1)
        spa_mask_1 = F.pad(spa_mask, ((self.k1 - 1) // 2,) * 4)
        spa_mask_1 = F.unfold(spa_mask_1, (self.k1, self.k1)).reshape(B, 1, self.k1 * self.k1, H // self.size,
                                                                      W // self.size).contiguous()
        print(spa_mask_1.shape, spa_kernel.shape)
        final_mask = (spa_kernel * spa_mask_1).sum(dim=2)
        ref = self.select_refinement_regions(final_mask).squeeze(dim=1)
        idx = torch.nonzero(ref, as_tuple=False)
        idx = idx[:, 0], idx[:, 1], idx[:, 2]  # P Hs Ws
        patches = self.crop_patch(out_lr, idx)  # P C size size
        return patches, idx, final_mask

if __name__ == '__main__':
    feature = torch.randn((2, 50, 64, 64))
    net = MPModule()
    patches, idx, final_mask = net(feature)
    print(patches.shape, final_mask.shape)
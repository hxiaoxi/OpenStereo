import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 2Dcostvolume 简洁版
def build_2Dcorr(img_left, img_right, max_disp=40, zero_volume=None):
    B, C, H, W = img_left.shape
    # if zero_volume is not None:
    # tmp_zero_volume = zero_volume  # * 0.0
    # print('tmp_zero_volume: ', mean)
    # volume = tmp_zero_volume
    # else:
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if (i > 0) & (i < W):
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, : W - i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume


# 3D costvolume
def build_3Dcorr(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1):
    """
    Concat left and right in Channel dimension to form the raw cost volume.
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index
        dilation (int): the step between near disparity index

    Inputs:
        reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
        target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

    Output:
        concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, disp_sample_number, Height, Width] layout

    """
    device = reference_fm.device
    N, C, H, W = reference_fm.shape

    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation
    disp_index = torch.linspace(start_disp, end_disp, disp_sample_number)

    concat_fm = torch.zeros(N, C * 2, disp_sample_number, H, W).to(device)
    idx = 0
    for i in disp_index:
        i = int(i)  # convert torch.Tensor to int, so that it can be index
        if i > 0:
            concat_fm[:, :C, idx, :, i:] = reference_fm[:, :, :, i:]
            concat_fm[:, C:, idx, :, i:] = target_fm[:, :, :, :-i]
        elif i == 0:
            concat_fm[:, :C, idx, :, :] = reference_fm
            concat_fm[:, C:, idx, :, :] = target_fm
        else:
            concat_fm[:, :C, idx, :, :i] = reference_fm[:, :, :, :i]
            concat_fm[:, C:, idx, :, :i] = target_fm[:, :, :, abs(i) :]
        idx = idx + 1

    concat_fm = concat_fm.contiguous()
    return concat_fm


def warp_right_to_left(x, disp, warp_grid=None):
    # print('size: ', x.size())

    B, C, H, W = x.size()
    # mesh grid
    if warp_grid is not None:
        xx0, yy = warp_grid
        xx = xx0 + disp
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
    else:
        # xx = torch.arange(0, W, device=disp.device).float()
        # yy = torch.arange(0, H, device=disp.device).float()
        xx = torch.arange(0, W, device=disp.device, dtype=x.dtype)
        yy = torch.arange(0, H, device=disp.device, dtype=x.dtype)
        # if x.is_cuda:
        #    xx = xx.cuda()
        #    yy = yy.cuda()
        xx = xx.view(1, -1).repeat(H, 1)
        yy = yy.view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        # apply disparity to x-axis
        xx = xx + disp
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
        yy = 2.0 * yy / max(H - 1, 1) - 1.0

    grid = torch.cat((xx, yy), 1)

    vgrid = grid
    # vgrid[:, 0, :, :] = vgrid[:, 0, :, :] + disp[:, 0, :, :]
    # vgrid[:, 0, :, :].add_(disp[:, 0, :, :])
    # vgrid.add_(disp)

    # scale grid to [-1,1]
    # vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:] / max(W-1,1)-1.0
    # vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:] / max(H-1,1)-1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    # mask = torch.autograd.Variable(torch.ones_like(x))
    # mask = nn.functional.grid_sample(mask, vgrid)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1

    # return output*mask
    return output  # *mask


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, pad=1, downsample=None):  # k_size=3
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu(out + x)
        return out


def make_layer(block, inplanes, outplanes, block_num, stride, pad):
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

    layers = [block(inplanes, outplanes, stride, pad, downsample)]
    for i in range(1, block_num):
        layers.append(block(outplanes, outplanes, 1, pad, None))

    return nn.Sequential(*layers)


class Encoder(nn.Module):  # 需要更新参数的网络层, 需要继承nn.Module
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=4, stride=2, pad=1, outpad=0, bias=False):
        super(Decoder, self).__init__()
        # k=2,s=2时, 为使Hout=2Hin, 需要outpad-2pad=0, 默认的pad和outpad都是0, 可以不额外设置
        # k=4,s=2时, 为使Hout=2Hin, 需要outpad-2pad+2=0, 可设置pad=1, outpad=0
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, output_padding=outpad, bias=bias)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


# common/modules.py有函数, x是否经过softmax激活?
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

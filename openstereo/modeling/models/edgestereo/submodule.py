import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DispCorrLayer(nn.Module):
    def __init__(self, max_disp=128, kernel_size=1, stride1=1, stride2=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        # assert stride1 == stride2 == 1
        super().__init__()
        self.max_disp = max_disp
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.padlayer = nn.ConstantPad2d((self.max_disp, 0, 0, 0), 0)  # 仅左侧补充pad

    def forward(self, in1, in2):  # input.shape N*C*H*W
        in2_pad = self.padlayer(in2)
        offsetx = torch.arange(0, self.max_disp, step=self.stride2)
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat([torch.mean(in1 * in2_pad[:, :, :, dx : dx + wid], 1, keepdim=True) for dx in offsetx], 1)
        # print('corr output:', output.shape)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):  # k_size=3,pad=k_size//2
        super().__init__()
        temp_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, temp_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(temp_channels)
        self.conv2 = nn.Conv2d(temp_channels, temp_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(temp_channels)
        self.conv3 = nn.Conv2d(temp_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:  # stride!=1, 输出分辨率下降, in_c!=out_c, 通道数变化
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def make_layer(block, in_channels, out_channels, num, stride=1):
    layers = []
    layers.append(block(in_channels, out_channels, stride))
    for _ in range(1, num):
        layers.append(block(out_channels, out_channels, 1))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, pad=1, downsample=None):  # k_size=3
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)
        return out


def _make_layer(self, block, planes, blocks, stride, pad, dilation):
    """
    block: class of basic net
    planes: out channels
    """
    downsample = None
    if stride != 1 or self.inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
    self.inplanes = planes
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

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


# common/modules.py有函数
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import pickle


# 我自己写的disp相关层
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


# 原版的flow相关层
class CorrTorch(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        # self.pad_size = pad_size
        # self.kernel_size = kernel_size
        # self.stride1 = stride1
        # self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):  # input.shape N*C*H*W
        in2_pad = self.padlayer(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1), torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat(
            [
                torch.mean(in1 * in2_pad[:, :, dy : dy + hei, dx : dx + wid], 1, keepdim=True)
                for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
            ],
            1,
        )
        return output
        # kernel_size锁死为1, 如果大于1, 就无法简单input1*input2
        # stride1不好改
        # stride2很好改，修改offsetx和offsety即可


def make_bilinear_weights(size, num_channels):
    """Generate bi-linear interpolation weights as up-sampling filters (following FCN paper)."""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5  ## 备注：center=(size-1)/2
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


class ResidualBlock(nn.Module):
    """BTNK1和BTNK2共用一个类, 用out_channels和stride区分, BTNK1的x需要downsample"""

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def conv_bn_relu(in_channels: int, out_channels: int, k=3, s=1):  # conv2d+bn+leakyrelu
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=(k - 1) // 2),
        # nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=(k-1)//2, bias=False),
        # nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )


def deconv_bn_relu(in_channel, out_channel, ksize=3, s=1):
    pad = (ksize - 1) // 2
    outpad = 2 * pad - 1
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=ksize, stride=s, padding=pad, output_padding=outpad),
        # nn.ConvTranspose2d(in_channel, out_channel, kernel_size=ksize, stride=s,
        #                    padding=pad, output_padding=outpad, bias=False),
        # nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(inplace=True),
    )


def make_layer(block, in_channels, out_channels, num, stride=1):
    layers = []
    layers.append(block(in_channels, out_channels, stride))
    for _ in range(1, num):
        layers.append(block(out_channels, out_channels, 1))
    return nn.Sequential(*layers)


class HED(nn.Module):
    """HED network."""

    def __init__(self):
        super(HED, self).__init__()
        # Layers.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights.
        # self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        # self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        # self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        # self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)
        # Register fixed bilinear weights as buffers
        self.register_buffer("weight_deconv2", make_bilinear_weights(4, 1))
        self.register_buffer("weight_deconv3", make_bilinear_weights(8, 1))
        self.register_buffer("weight_deconv4", make_bilinear_weights(16, 1))
        self.register_buffer("weight_deconv5", make_bilinear_weights(32, 1))

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = self.prepare_aligned_crop()

    def load_checkpoint(self, opt=None, path='./checkpoint.pth'):
        """ Load previous pre-trained checkpoint.
        :param net:  Network instance.
        :param opt:  Optimizer instance.
        :param path: Path of checkpoint file.
        :return:     Checkpoint epoch number.
        """
        # print(path)
        # D:\Code\EdgeStereo\xwj_hed\data\hed_checkpoint.pt
        if os.path.isfile(path):
            print('=> Loading checkpoint {}...'.format(path))
            checkpoint = torch.load(path)

            new_state_dict = {}
            # print(checkpoint['net'].keys())
            for key, value in checkpoint['net'].items():
                # 去掉参数名中的 'module.' 前缀
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            # print(new_state_dict.keys())
            # print(self.state_dict().keys())
            self.load_state_dict(new_state_dict,strict=False)
            # net.load_state_dict(checkpoint['net'])
            # opt.load_state_dict(checkpoint['opt'])
            return checkpoint['epoch']
        else:
            raise ValueError('=> No checkpoint found at {}.'.format(path))

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """Prepare for aligned crop."""
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """Mapping inverse."""
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """Mapping compose."""
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """Deconvolution coordinates mapping."""
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """Convolution coordinates mapping."""
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """Pooling coordinates mapping."""
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, x):
        # VGG-16 network.
        image_h, image_w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)
        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        # 反卷积到score_dsn1的size, 即原分辨率
        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        crop1 = score_dsn1[:, :, self.crop1_margin : self.crop1_margin + image_h, self.crop1_margin : self.crop1_margin + image_w]
        crop2 = upsample2[:, :, self.crop2_margin : self.crop2_margin + image_h, self.crop2_margin : self.crop2_margin + image_w]
        crop3 = upsample3[:, :, self.crop3_margin : self.crop3_margin + image_h, self.crop3_margin : self.crop3_margin + image_w]
        crop4 = upsample4[:, :, self.crop4_margin : self.crop4_margin + image_h, self.crop4_margin : self.crop4_margin + image_w]
        crop5 = upsample5[:, :, self.crop5_margin : self.crop5_margin + image_h, self.crop5_margin : self.crop5_margin + image_w]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
        results = [crop1, crop2, crop3, crop4, crop5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results


class Encoder(nn.Module):  # 需要更新参数的定义为继承nn.Module的类
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        # k=2,s=2时, 为使Hout=2Hin, 需要outpad-2pad=0, 默认2个pad都是0
        # k=4,s=2时, 为使Hout=2Hin, 需要outpad-2pad+2=0, 那可否设置为pad=1,outpad=0
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class disp_regression(nn.Module):  # 不需要更新参数的可以定义为def
    def __init__(self, maxdisp):
        super().__init__()
        disp = torch.FloatTensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]))
        self.register_buffer("disp", disp)

    def forward(self, x):  # x是概率
        # disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3]) # 不用repeat也可以按位相乘
        out = torch.sum(x * self.disp, 1)  # 按位相乘并求和, 所有视差和其概率相乘并求和
        return out  # N*1*H*W


def disp_reg(x: torch.Tensor, maxdisp: int) -> torch.Tensor:
    disp = torch.FloatTensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()
    out = torch.sum(x * disp, 1)  # 按位相乘并求和, 所有视差和其概率相乘并求和, 加权和
    return out


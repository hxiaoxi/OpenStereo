import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from modeling.common.basic_layers import conv3d_bn_relu, conv3d_bn, deconv3d_bn

import scipy.io as sio



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
        self.register_buffer("weight_deconv2", self.make_bilinear_weights(4, 1))
        self.register_buffer("weight_deconv3", self.make_bilinear_weights(8, 1))
        self.register_buffer("weight_deconv4", self.make_bilinear_weights(16, 1))
        self.register_buffer("weight_deconv5", self.make_bilinear_weights(32, 1))

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = self.prepare_aligned_crop()

    def load_checkpoint(self, path="./checkpoint.pth"):
        """Load previous pre-trained checkpoint.
        :param path: Path of checkpoint file.
        :return:     Checkpoint epoch number.
        """
        if os.path.isfile(path):
            print("=> Loading checkpoint {}...".format(path))
            checkpoint = torch.load(path)

            new_state_dict = {}
            # print(checkpoint['net'].keys())
            for key, value in checkpoint.items():
                # 去掉参数名中的 'module.' 前缀
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            # print(new_state_dict.keys())
            # print(self.state_dict().keys())
            self.load_state_dict(new_state_dict, strict=False)  # 模型缺少插值权重weight_deconv
            return checkpoint["epoch"]
        else:
            raise ValueError("=> No checkpoint found at {}.".format(path))

    # HED使用该函数生成插值权重
    def make_bilinear_weights(self, size, num_channels):
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


class RCF(nn.Module):
    def __init__(self, pretrained=None):
        super(RCF, self).__init__()
        self.conv1_1 = nn.Conv2d(  3,  64, 3, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.act = nn.ReLU(inplace=True)

        self.conv1_1_down = nn.Conv2d( 64, 21, 1)
        self.conv1_2_down = nn.Conv2d( 64, 21, 1)
        self.conv2_1_down = nn.Conv2d(128, 21, 1)
        self.conv2_2_down = nn.Conv2d(128, 21, 1)
        self.conv3_1_down = nn.Conv2d(256, 21, 1)
        self.conv3_2_down = nn.Conv2d(256, 21, 1)
        self.conv3_3_down = nn.Conv2d(256, 21, 1)
        self.conv4_1_down = nn.Conv2d(512, 21, 1)
        self.conv4_2_down = nn.Conv2d(512, 21, 1)
        self.conv4_3_down = nn.Conv2d(512, 21, 1)
        self.conv5_1_down = nn.Conv2d(512, 21, 1)
        self.conv5_2_down = nn.Conv2d(512, 21, 1)
        self.conv5_3_down = nn.Conv2d(512, 21, 1)

        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_fuse = nn.Conv2d(5, 1, 1)

        self.weight_deconv2 = self._make_bilinear_weights( 4, 1).cuda()
        self.weight_deconv3 = self._make_bilinear_weights( 8, 1).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(16, 1).cuda()

        # init weights
        self.apply(self._init_weights)
        if pretrained is not None:
            vgg16 = sio.loadmat(pretrained)
            torch_params = self.state_dict()

            for k in vgg16.keys():
                name_par = k.split('-')
                size = len(name_par)
                if size == 2:
                    name_space = name_par[0] + '.' + name_par[1]
                    data = np.squeeze(vgg16[k])
                    torch_params[name_space] = torch.from_numpy(data)
            self.load_state_dict(torch_params)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                nn.init.constant_(m.weight, 0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Based on HED implementation @ https://github.com/xwjabc/hed
    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

    # Based on BDCN implementation @ https://github.com/pkuCactus/BDCN
    def _crop(self, data, img_h, img_w, crop_h, crop_w):
        _, _, h, w = data.size()
        assert(img_h <= h and img_w <= w)
        data = data[:, :, crop_h:crop_h + img_h, crop_w:crop_w + img_w]
        return data

    def load_checkpoint(self, path="./checkpoint.pth"):
        """Load previous pre-trained checkpoint.
        :param path: Path of checkpoint file.
        :return:     Checkpoint epoch number.
        """
        if os.path.isfile(path):
            print("=> Loading checkpoint {}...".format(path))
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint)
            # self.load_state_dict(checkpoint, strict=False)  # 模型缺少插值权重weight_deconv
            print("=> checkpoint loaded")
        else:
            raise ValueError("=> No checkpoint found at {}.".format(path))

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]
        conv1_1 = self.act(self.conv1_1(x))
        conv1_2 = self.act(self.conv1_2(conv1_1))
        pool1   = self.pool1(conv1_2)
        conv2_1 = self.act(self.conv2_1(pool1))
        conv2_2 = self.act(self.conv2_2(conv2_1))
        pool2   = self.pool2(conv2_2)
        conv3_1 = self.act(self.conv3_1(pool2))
        conv3_2 = self.act(self.conv3_2(conv3_1))
        conv3_3 = self.act(self.conv3_3(conv3_2))
        pool3   = self.pool3(conv3_3)
        conv4_1 = self.act(self.conv4_1(pool3))
        conv4_2 = self.act(self.conv4_2(conv4_1))
        conv4_3 = self.act(self.conv4_3(conv4_2))
        pool4   = self.pool4(conv4_3)
        conv5_1 = self.act(self.conv5_1(pool4))
        conv5_2 = self.act(self.conv5_2(conv5_1))
        conv5_3 = self.act(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        out1 = self.score_dsn1(conv1_1_down + conv1_2_down)
        out2 = self.score_dsn2(conv2_1_down + conv2_2_down)
        out3 = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        out4 = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        out5 = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        out2 = F.conv_transpose2d(out2, self.weight_deconv2, stride=2)
        out3 = F.conv_transpose2d(out3, self.weight_deconv3, stride=4)
        out4 = F.conv_transpose2d(out4, self.weight_deconv4, stride=8)
        out5 = F.conv_transpose2d(out5, self.weight_deconv5, stride=8)

        out2 = self._crop(out2, img_h, img_w, 1, 1)
        out3 = self._crop(out3, img_h, img_w, 2, 2)
        out4 = self._crop(out4, img_h, img_w, 4, 4)
        out5 = self._crop(out5, img_h, img_w, 0, 0)

        fuse = torch.cat((out1, out2, out3, out4, out5), dim=1)
        fuse = self.score_fuse(fuse)
        results = [out1, out2, out3, out4, out5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .edgenet import HED as edgeNet
from modeling.common.basic_layers import conv3d_bn_relu, conv3d_bn, deconv3d_bn


def cat_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1):
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


class Hourglass(nn.Module):
    """
    An implementation of hourglass module proposed in PSMNet.
    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer,
            default True
    Inputs:
        x, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        presqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        postsqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
    Outputs:
        out, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        pre, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        post, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout

    """

    def __init__(self, in_planes, batch_norm=True):
        super(Hourglass, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = conv3d_bn_relu(self.batch_norm, in_planes, in_planes * 2, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv2 = conv3d_bn(self.batch_norm, in_planes * 2, in_planes * 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = conv3d_bn_relu(self.batch_norm, in_planes * 2, in_planes * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = conv3d_bn_relu(self.batch_norm, in_planes * 2, in_planes * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = deconv3d_bn(self.batch_norm, in_planes * 2, in_planes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)
        self.conv6 = deconv3d_bn(self.batch_norm, in_planes * 2, in_planes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

    def forward(self, x, presqu=None, postsqu=None):
        # in: [B, C, D, H, W], out: [B, 2C, D, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/2, W/2]
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/4, W/4]
        out = self.conv4(out)

        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/2, W/2]
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        out = self.conv6(post)

        return out, pre, post


class PSMAggregator(nn.Module):
    """
    Args:
        max_disp (int): max disparity
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): concatenation-based cost volume without further processing,
            in [BatchSize, in_planes, MaxDisparity//4, Height//4, Width//4] layout
    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout
    """

    def __init__(self, max_disp, in_planes=64, batch_norm=True):
        super(PSMAggregator, self).__init__()
        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.in_planes, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
        )
        self.dres1 = nn.Sequential(conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False), conv3d_bn(batch_norm, 32, 32, 3, 1, 1, bias=False))
        self.dres2 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres3 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres4 = Hourglass(in_planes=32, batch_norm=batch_norm)

        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False), nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, raw_cost):
        B, C, D, H, W = raw_cost.shape
        # raw_cost: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        cost0 = self.dres0(raw_cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre2, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        # (BatchSize, 1, max_disp, Height, Width)
        full_h, full_w = H * 4, W * 4
        align_corners = True
        cost1 = F.interpolate(cost1, [self.max_disp, full_h, full_w], mode="trilinear", align_corners=align_corners)
        cost2 = F.interpolate(cost2, [self.max_disp, full_h, full_w], mode="trilinear", align_corners=align_corners)
        cost3 = F.interpolate(cost3, [self.max_disp, full_h, full_w], mode="trilinear", align_corners=align_corners)

        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
        cost2 = torch.squeeze(cost2, 1)
        cost3 = torch.squeeze(cost3, 1)

        return [cost3, cost2, cost1]


class EdgePSMCostProcessor(nn.Module):
    def __init__(self, max_disp=192, in_planes=76):
        super().__init__()
        self.cat_func = partial(
            cat_fms,
            max_disp=int(max_disp // 4),
            start_disp=0,
            dilation=1,
        )
        self.aggregator = PSMAggregator(max_disp=max_disp, in_planes=in_planes)
        self.edgeNet = edgeNet()
        # HED直接导入训练好的参数并固定参数
        # self.edgeNet.load_checkpoint(path="/home/huangjx/Documents/OpenStereo/hed_checkpoint.pt") # linux位置
        self.edgeNet.load_checkpoint(path="D:/Code/OpenStereo/hed_checkpoint.pt") # windows位置
        for name, param in self.edgeNet.named_parameters():
            param.requires_grad = False

    def forward(self, inputs):
        # 1. build raw cost by concat
        left_feature = inputs["ref_feature"]
        right_feature = inputs["tgt_feature"]

        left_img = inputs["ref_img"]
        right_img = inputs["tgt_img"]
        # HED output: sigmoid([crop1, crop2, crop3, crop4, crop5, fuse])
        left_edge = self.edgeNet(left_img)
        right_edge = self.edgeNet(right_img)
        left_edge = torch.cat(left_edge, dim=1)  # N*6*H*W
        right_edge = torch.cat(right_edge, dim=1)
        _, _, H, W = left_feature.shape
        left_edge = F.interpolate(left_edge, [H, W], mode="bilinear", align_corners=True)
        right_edge = F.interpolate(right_edge, [H, W], mode="bilinear", align_corners=True)

        # 核心增加内容, 拼接边缘特征
        left_feature = torch.cat([left_feature, left_edge], dim=1)
        right_feature = torch.cat([right_feature, right_edge], dim=1)

        cat_cost = self.cat_func(left_feature, right_feature)
        # 2. aggregate cost by 3D-hourglass
        costs = self.aggregator(cat_cost)
        [cost3, cost2, cost1] = costs
        return {"cost1": cost1, "cost2": cost2, "cost3": cost3}

    def input_output(self):
        return {"inputs": ["ref_feature", "tgt_feature"], "outputs": ["cost1", "cost2", "cost3"]}


# class HED(nn.Module):
#     """HED network."""

#     def __init__(self):
#         super(HED, self).__init__()
#         # Layers.
#         self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
#         self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

#         self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

#         self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

#         self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

#         self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

#         self.relu = nn.ReLU()
#         # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
#         #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
#         #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
#         #       maps will possibly be smaller than the original images.
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

#         self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
#         self.score_dsn2 = nn.Conv2d(128, 1, 1)
#         self.score_dsn3 = nn.Conv2d(256, 1, 1)
#         self.score_dsn4 = nn.Conv2d(512, 1, 1)
#         self.score_dsn5 = nn.Conv2d(512, 1, 1)
#         self.score_final = nn.Conv2d(5, 1, 1)

#         # Fixed bilinear weights.
#         # self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
#         # self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
#         # self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
#         # self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)
#         # Register fixed bilinear weights as buffers
#         self.register_buffer("weight_deconv2", self.make_bilinear_weights(4, 1))
#         self.register_buffer("weight_deconv3", self.make_bilinear_weights(8, 1))
#         self.register_buffer("weight_deconv4", self.make_bilinear_weights(16, 1))
#         self.register_buffer("weight_deconv5", self.make_bilinear_weights(32, 1))

#         # Prepare for aligned crop.
#         self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = self.prepare_aligned_crop()

#     def load_checkpoint(self, path="./checkpoint.pth"):
#         """Load previous pre-trained checkpoint.
#         :param path: Path of checkpoint file.
#         :return:     Checkpoint epoch number.
#         """
#         if os.path.isfile(path):
#             print("=> Loading checkpoint {}...".format(path))
#             checkpoint = torch.load(path)

#             new_state_dict = {}
#             # print(checkpoint['net'].keys())
#             for key, value in checkpoint["net"].items():
#                 # 去掉参数名中的 'module.' 前缀
#                 new_key = key.replace("module.", "")
#                 new_state_dict[new_key] = value
#             # print(new_state_dict.keys())
#             # print(self.state_dict().keys())
#             self.load_state_dict(new_state_dict, strict=False)  # 模型缺少插值权重weight_deconv
#             return checkpoint["epoch"]
#         else:
#             raise ValueError("=> No checkpoint found at {}.".format(path))

#     # HED使用该函数生成插值权重
#     def make_bilinear_weights(self, size, num_channels):
#         """Generate bi-linear interpolation weights as up-sampling filters (following FCN paper)."""
#         factor = (size + 1) // 2
#         if size % 2 == 1:
#             center = factor - 1
#         else:
#             center = factor - 0.5  ## 备注：center=(size-1)/2
#         og = np.ogrid[:size, :size]
#         filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
#         filt = torch.from_numpy(filt)
#         w = torch.zeros(num_channels, num_channels, size, size)
#         w.requires_grad = False  # Set not trainable.
#         for i in range(num_channels):
#             for j in range(num_channels):
#                 if i == j:
#                     w[i, j] = filt
#         return w

#     # noinspection PyMethodMayBeStatic
#     def prepare_aligned_crop(self):
#         """Prepare for aligned crop."""
#         # Re-implement the logic in deploy.prototxt and
#         #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
#         # Other reference materials:
#         #   hed/include/caffe/layer.hpp
#         #   hed/include/caffe/vision_layers.hpp
#         #   hed/include/caffe/util/coords.hpp
#         #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

#         def map_inv(m):
#             """Mapping inverse."""
#             a, b = m
#             return 1 / a, -b / a

#         def map_compose(m1, m2):
#             """Mapping compose."""
#             a1, b1 = m1
#             a2, b2 = m2
#             return a1 * a2, a1 * b2 + b1

#         def deconv_map(kernel_h, stride_h, pad_h):
#             """Deconvolution coordinates mapping."""
#             return stride_h, (kernel_h - 1) / 2 - pad_h

#         def conv_map(kernel_h, stride_h, pad_h):
#             """Convolution coordinates mapping."""
#             return map_inv(deconv_map(kernel_h, stride_h, pad_h))

#         def pool_map(kernel_h, stride_h, pad_h):
#             """Pooling coordinates mapping."""
#             return conv_map(kernel_h, stride_h, pad_h)

#         x_map = (1, 0)
#         conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
#         conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
#         pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

#         conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
#         conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
#         pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

#         conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
#         conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
#         conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
#         pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

#         conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
#         conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
#         conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
#         pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

#         conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
#         conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
#         conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

#         score_dsn1_map = conv1_2_map
#         score_dsn2_map = conv2_2_map
#         score_dsn3_map = conv3_3_map
#         score_dsn4_map = conv4_3_map
#         score_dsn5_map = conv5_3_map

#         upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
#         upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
#         upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
#         upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

#         crop1_margin = int(score_dsn1_map[1])
#         crop2_margin = int(upsample2_map[1])
#         crop3_margin = int(upsample3_map[1])
#         crop4_margin = int(upsample4_map[1])
#         crop5_margin = int(upsample5_map[1])

#         return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

#     def forward(self, x):
#         # VGG-16 network.
#         image_h, image_w = x.shape[2], x.shape[3]
#         conv1_1 = self.relu(self.conv1_1(x))
#         conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
#         pool1 = self.maxpool(conv1_2)

#         conv2_1 = self.relu(self.conv2_1(pool1))
#         conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
#         pool2 = self.maxpool(conv2_2)

#         conv3_1 = self.relu(self.conv3_1(pool2))
#         conv3_2 = self.relu(self.conv3_2(conv3_1))
#         conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
#         pool3 = self.maxpool(conv3_3)

#         conv4_1 = self.relu(self.conv4_1(pool3))
#         conv4_2 = self.relu(self.conv4_2(conv4_1))
#         conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
#         pool4 = self.maxpool(conv4_3)

#         conv5_1 = self.relu(self.conv5_1(pool4))
#         conv5_2 = self.relu(self.conv5_2(conv5_1))
#         conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

#         score_dsn1 = self.score_dsn1(conv1_2)
#         score_dsn2 = self.score_dsn2(conv2_2)
#         score_dsn3 = self.score_dsn3(conv3_3)
#         score_dsn4 = self.score_dsn4(conv4_3)
#         score_dsn5 = self.score_dsn5(conv5_3)

#         # 反卷积到score_dsn1的size, 即原分辨率
#         upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
#         upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
#         upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
#         upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

#         # Aligned cropping.
#         crop1 = score_dsn1[:, :, self.crop1_margin : self.crop1_margin + image_h, self.crop1_margin : self.crop1_margin + image_w]
#         crop2 = upsample2[:, :, self.crop2_margin : self.crop2_margin + image_h, self.crop2_margin : self.crop2_margin + image_w]
#         crop3 = upsample3[:, :, self.crop3_margin : self.crop3_margin + image_h, self.crop3_margin : self.crop3_margin + image_w]
#         crop4 = upsample4[:, :, self.crop4_margin : self.crop4_margin + image_h, self.crop4_margin : self.crop4_margin + image_w]
#         crop5 = upsample5[:, :, self.crop5_margin : self.crop5_margin + image_h, self.crop5_margin : self.crop5_margin + image_w]

#         # Concatenate according to channels.
#         fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
#         fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
#         results = [crop1, crop2, crop3, crop4, crop5, fuse]
#         results = [torch.sigmoid(r) for r in results]
#         return results

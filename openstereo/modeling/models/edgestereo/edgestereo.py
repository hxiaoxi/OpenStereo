import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.optim as optim
from .submodule import *
from modeling.common.basic_layers import conv_bn, conv_bn_relu, deconv_bn_relu, BasicBlock


class edgestereo(nn.Module):
    def __init__(self, model_cfg, *args, **kwargs):
        super(edgestereo, self).__init__(*args, **kwargs)

        # 在yaml中设置base_config
        self.max_disp = model_cfg["base_config"]["max_disp"]

        # edge dection
        self.bn = model_cfg["base_config"]["batch_norm"]
        self.conv1_x = nn.Sequential(  # resnet-50的stage0, 用3个3*3卷积替代原本的1个7*7卷积
            conv_bn_relu(self.bn, 3, 64, 3, 1, 1, 1, bias=False),
            conv_bn_relu(self.bn, 64, 64, 3, 1, 1, 1, bias=False),
            conv_bn_relu(self.bn, 64, 128, 3, 1, 1, 1, bias=False),
        )  # output.shape:B*128*H*W
        # self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.HEDNet = HED()
        # HED直接导入训练好的参数并固定参数
        self.HEDNet.load_checkpoint(path=model_cfg["base_config"]["hedpt_path"])
        for name, param in self.HEDNet.named_parameters():
            param.requires_grad = False

        self.corr_layer = DispCorrLayer(max_disp=self.max_disp)

        self.chs = self.max_disp + 6  # corr + edge
        # left_feature刚经过corr得到corr_cost就又和left_feature拼接, 看起来没有什么意义
        # 但maxD=192, 和edge的6通道相比, 差两个数量级, 可能会影响效果

        self.encoder1 = Encoder(self.chs, 256)  # 1/2
        self.encoder2 = Encoder(256, 512)  # 1/4
        self.encoder3 = Encoder(512, 1024)  # 1/8
        # self.encoder4 = Encoder(1024, 2048)
        # self.res_encoder1 = make_layer(ResidualBlock, self.chs, 256, 3, 2)  # 1/2
        # self.res_encoder2 = make_layer(ResidualBlock, 256, 512, 4, 2)  # 1/4
        # self.res_encoder3 = make_layer(ResidualBlock, 512, 1024, 6, 2)  # 1/8
        # self.res_encoder4 = make_layer(ResidualBlock, 1024, 1024, 3, 2)  # 1/16
        # self.conv_ende = conv_bn_relu(1024, 1024, 3, 1)

        # self.decoder4 = Decoder(2048, 1024)
        self.decoder3 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder1 = Decoder(256, 128)
        # skip connection
        # self.decoder4 = deconv_bn_relu(1024+1024+32, 1024, ksize=3, s=2)
        # self.decoder3 = deconv_bn_relu(1024+1024+32, 512, ksize=3, s=2)
        # self.decoder2 = deconv_bn_relu(512+512+32, 256, ksize=3, s=2)
        # self.decoder1 = deconv_bn_relu(256+256+32, 128, ksize=3, s=2)
        # no skip connection
        # self.decoder4 = deconv_bn_relu(1024, 1024, ksize=3, s=2)  # 1/8
        # self.decoder3 = deconv_bn_relu(1024, 512, ksize=3, s=2)  # 1/4
        # self.decoder2 = deconv_bn_relu(512, 256, ksize=3, s=2)  # 1/2
        # self.decoder1 = deconv_bn_relu(256, 128, ksize=3, s=2)  # 1

        self.last_conv = nn.Sequential(
            conv_bn_relu(self.bn, 128, 64, 3, 1, 1, 1, bias=False),
            conv_bn_relu(self.bn, 64, 32, 3, 1, 1, 1, bias=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )
        # baseTrainer里有模型参数初始化, 可在yaml中设置

    def forward(self, inputs):
        # inputs keys: dict_keys(['ref_img', 'tgt_img', 'disp_gt', 'mask', 'index'])
        left_img = inputs["ref_img"]
        right_img = inputs["tgt_img"]

        # edge detection, HED的返回内容如下
        # fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        # fuse = self.score_final(fuse_cat)  # Shape: B*5*H*W -> B*1*H*W
        # results = [crop1, crop2, crop3, crop4, crop5, fuse]
        # results = [torch.sigmoid(r) for r in results]
        pre_edges = self.HEDNet(left_img)  # 原图大小的边缘

        # corr layer
        left_1 = self.conv1_x(left_img)  # 直接在第一步 left_img + edge 可以吗, 有必要吗?
        right_1 = self.conv1_x(right_img)
        corr_fea = self.corr_layer(left_1, right_1)  # b*maxD*H*W
        # 应该在cost volume后加权和估计初始视差吗?
        # init_disparity = weight_softmax(corr_fea);
        cat_edge = torch.cat(pre_edges, dim=1)  # b*6*H*W
        hybrid_fea = torch.cat([corr_fea, cat_edge], dim=1)  # b*(maxD+6)*H*W

        # encoder:resnet50:3-4-6-3
        # enc1 = self.res_encoder1(hybrid_fea)  # 1/2
        # enc2 = self.res_encoder2(enc1)  # 1/4
        # enc3 = self.res_encoder3(enc2)  # 1/8
        # enc4 = self.res_encoder4(enc3)  # 1/16
        enc1 = self.encoder1(hybrid_fea)  # 256 * 1/2
        enc2 = self.encoder2(enc1)  # 512 * 1/4
        enc3 = self.encoder3(enc2)  # 1024 * 1/8
        # enc4 = self.encoder4(enc3)  # 2048 * 1/16

        # decoder:
        # dec4 = torch.cat([dec4, enc4, e4], dim=1)
        # dec3 = self.decoder4(dec4)  # 1024 * 1/8
        # dec3 = torch.cat([dec3, enc3, e3], dim=1)
        # dec2 = self.decoder3(dec3)  # 512 * 1/4
        # dec2 = torch.cat([dec2, enc2, e2], dim=1)
        # dec1 = self.decoder2(dec2)  # 256 * 1/2
        # dec1 = torch.cat([dec1, enc1, e1], dim=1)
        # output = self.decoder1(dec1)  # 128 * 1

        # if self.decoder_type='cat'or'add'
        # dec = self.decoder4(enc4)  # 1024 * 1/8
        dec2 = self.decoder3(enc3)  # 512 * 1/4
        dec1 = self.decoder2(dec2)  # 256 * 1/2
        output = self.decoder1(dec1)  # 128 * 1

        # 最终的激活方式感觉有问题. 如何激活得到[0,max_disp]范围的数值
        disparity = self.last_conv(output)  # 64->32->1,最后一层没有bn和relu
        disparity = torch.clamp(disparity, min=0, max=self.max_disp)
        # disparity = torch.sigmoid(disparity) * self.max_disp
        # disparity = F.relu(disparity, inplace=True) # 去除负值视差

        # softmax?

        return disparity, pre_edges[-1]


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
            for key, value in checkpoint["net"].items():
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.optim as optim
from .submodule import *


class edgestereo(nn.Module):
    def __init__(self, model_cfg, *args, **kwargs):
        super(edgestereo, self).__init__(*args, **kwargs)

        # 在yaml中设置base_config
        self.max_disp = model_cfg["base_config"]["max_disp"]
        # self.num_downsample = model_cfg['base_config']['num_downsample']
        # self.num_scales = model_cfg['base_config']['num_scales']

        # edge dection
        self.conv1_x = nn.Sequential(  # resnet-50的stage0, 用3个3*3卷积替代原本的1个7*7卷积
            conv_bn_relu(3, 64, k=3, s=1), conv_bn_relu(64, 64, k=3, s=1), conv_bn_relu(64, 128, k=3, s=1)
        )  # size不变, output:b*128*H*W
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.HEDNet = HED()
        # HED直接导入训练好的参数
        self.HEDNet.load_checkpoint(path=model_cfg["base_config"]["hedpt_path"])
        # 并固定参数
        for name, param in self.HEDNet.named_parameters():
            param.requires_grad = False

        self.corr_layer = DispCorrLayer(max_disp=self.max_disp)

        self.chs = self.max_disp + 128 + 6  # corr + conv_1 + edge
        self.encoder1 = Encoder(self.chs, 256)  # 1/2
        self.encoder2 = Encoder(256, 512)  # 1/4
        self.encoder3 = Encoder(512, 1024)  # 1/8
        self.encoder4 = Encoder(1024, 2048)
        # self.res_encoder1 = make_layer(ResidualBlock, self.chs, 256, 3, 2)  # 1/2
        # self.res_encoder2 = make_layer(ResidualBlock, 256, 512, 4, 2)  # 1/4
        # self.res_encoder3 = make_layer(ResidualBlock, 512, 1024, 6, 2)  # 1/8
        # self.res_encoder4 = make_layer(ResidualBlock, 1024, 1024, 3, 2)  # 1/16
        # self.conv_ende = conv_bn_relu(1024, 1024, 3, 1)

        self.decoder4 = Decoder(2048, 1024)
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
            conv_bn_relu(128, 64, k=3, s=1),
            conv_bn_relu(64, 32, k=3, s=1),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )
        # print(self.state_dict().keys())
        self.initialize_weights()  # relu和leakyrelu不统一, 效果存疑

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # basemodel里有参数初始化
    # def init_parameters(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
    #             nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias.data, 0.0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias.data, 0.0)
    #         elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
    #             if m.affine:
    #                 nn.init.normal_(m.weight.data, 1.0, 0.02)
    #                 nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        # print('inputs keys:',inputs.keys())
        # dict_keys(['ref_img', 'tgt_img', 'disp_gt', 'mask', 'index'])
        left_img = inputs["ref_img"]
        right_img = inputs["tgt_img"]

        # edge detection
        pre_edges = self.HEDNet(left_img)
        # results = [crop1, crop2, crop3, crop4, crop5, fuse] # shape:5*1*H*W
        # results = [torch.sigmoid(r) for r in results]
        # if train_edge:
        #     return pre_edges[5]  # return sigmoid(fuse)

        # corr layer
        left_1 = self.conv1_x(left_img)  # 1
        right_1 = self.conv1_x(right_img)
        corr_fea = self.corr_layer(left_1, right_1)  # b*maxD*H*W
        # init_disparity = weight_softmax(corr_fea);
        cat_edge = torch.cat(pre_edges, dim=1)  # b*6*H*W
        hybrid_fea = torch.cat([left_1, corr_fea, cat_edge], dim=1)

        # encoder:resnet50:3-4-6-3
        # enc1 = self.res_encoder1(hybrid_fea)  # 1/2
        # enc2 = self.res_encoder2(enc1)  # 1/4
        # enc3 = self.res_encoder3(enc2)  # 1/8
        # enc4 = self.res_encoder4(enc3)  # 1/16
        enc1 = self.encoder1(hybrid_fea)  # 256 * 1/2
        enc2 = self.encoder2(enc1)  # 512 * 1/4
        enc3 = self.encoder3(enc2)  # 1024 * 1/8
        enc4 = self.encoder4(enc3)  # 2048 * 1/16

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
        dec = self.decoder4(enc4)  # 1024 * 1/8
        dec = self.decoder3(dec)  # 512 * 1/4
        dec = self.decoder2(dec)  # 256 * 1/2
        output = self.decoder1(dec)  # 128 * 1

        # 最终的激活方式感觉有问题. 如何激活得到[0,max_disp]范围的数值
        disparity = self.last_conv(output)  # 64->32->1,最后一层没有bn和relu
        disparity = torch.sigmoid(disparity) * self.max_disp
        disparity = F.relu(disparity, inplace=True)

        return disparity
        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
        cost_volume = self.cost_volume_construction(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])
        return disparity_pyramid

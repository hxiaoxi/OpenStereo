import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .edgestereo import edgestereo
from modeling.base_model import BaseModel
from base_trainer import BaseTrainer
from utils import get_attr_from, get_valid_args
from modeling.common.lamb import Lamb  # 优化器, 继承自Optimizer


class EdgeStereo(BaseModel):
    def __init__(self, *args, **kwargs):
        super(EdgeStereo, self).__init__(*args, **kwargs)
        self.Trainer = EdgeStereo_Trainer

    def build_network(self):
        # backbone_cfg
        # cost_processor_cfg
        # disp_processor_cfg
        self.net = edgestereo(model_cfg=self.model_cfg)  # self.model_cfg在BaseModel中定义

        # 基类通过以下语句控制是否生成backbone, 不需要使用的话yaml中注释掉对应的cfg
        # if 'backbone_cfg' in model_cfg.keys():
        # if 'cost_processor_cfg' in model_cfg.keys():
        # if 'disp_processor_cfg' in model_cfg.keys():

    def forward(self, inputs):
        # input的格式是词典, 需要修改net的forward的参数
        # inputs.keys(): dict_keys(['ref_img', 'tgt_img', 'disp_gt', 'mask', 'index'])
        # ref_img = inputs["ref_img"]
        # tgt_img = inputs["tgt_img"]
        disparity_map = self.net(inputs)

        # print(disparity_map.shape)
        if disparity_map.dim()==4 and disparity_map.shape[1]==1:
            disparity_map=disparity_map.squeeze(1)
        if self.training:
            # 3层词典嵌套
            output = {
                "training_disp": {
                    # disparity_map 需要和 yaml 中 loss_cfg/log_prefix 参数保持一致
                    "disparity_map": {
                        "disp_ests": [disparity_map],
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"],
                    },
                    # 如果还有多个loss计算，可以传递其他log-prefix（猜测，应该成立，在yaml定义其他的loss计算方式）
                    # "another disparity_map": {
                    #     "disp_ests": disparity_map,
                    #     "disp_gt": inputs["disp_gt"],
                    #     "mask": inputs["mask"],
                    # },
                },
                "visual_summary": {},
                # "visual_summary": {
                #     'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                #     'image/train/disp_c': torch.cat([inputs['disp_gt'][0], res[-1][0]], dim=0),
                # },
            }
        else:
            # 因为是val或test, 所以一个batch只取一张图像返回吗?
            # test数据集的inputs.keys(): dict_keys(['ref_img', 'tgt_img', 'name', 'index'])

            if disparity_map[-1].dim() == 4:  # shape为B*1*H*W, 删除通道为 B*H*W
                disparity_map[-1] = disparity_map[-1].squeeze(1)
            # cated_image = torch.cat((disparity_map[-1], inputs["ref_img"][0, 0].unsqueeze(0)), dim=2)  # shape:B*H*W
            output = {
                "inference_disp": {
                    # disp_est而不是ests, 单数而非复数, 只返回一张图
                    "disp_est": disparity_map[-1],
                    # "ref_img" : inputs["ref_img"],
                    # "disp_gt": inputs["disp_gt"],
                    # "mask": inputs["mask"],
                },
                "visual_summary": {},
                # "visual_summary": {
                #     'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                #     'image/test/disp_c': res[0][0][0],
                # }
            }
        return output


class EdgeStereo_Trainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module = None,
        trainer_cfg: dict = None,
        data_cfg: dict = None,
        is_dist: bool = True,
        rank: int = None,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        super(EdgeStereo_Trainer, self).__init__(model, trainer_cfg, data_cfg, is_dist, rank, device, **kwargs)

    def build_optimizer(self, optimizer_cfg):
        if optimizer_cfg["solver"] == "lamb":  # 其他有用RMSprop或Adam, 出于什么考虑, 我用简单的Adam就好了吧
            # use lamb optimizer
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = Lamb(filter(lambda p: p.requires_grad, self.model.parameters()), lr=optimizer_cfg["lr"])
        else:
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = get_attr_from([optim], optimizer_cfg["solver"])
            valid_arg = get_valid_args(optimizer, optimizer_cfg, ["solver"])
            optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer = optimizer
        # 基类的build_optimizer函数, 只有else的部分
        # self.msg_mgr.log_info(optimizer_cfg)
        # optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        # valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        # optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        # self.optimizer = optimizer

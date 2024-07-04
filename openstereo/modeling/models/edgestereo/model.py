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
        # self.Trainer = EdgeStereo_Trainer

    def build_network(self):
        # self.model_cfg在BaseModel中定义
        self.net = edgestereo(model_cfg=self.model_cfg)
        # 基类通过以下语句控制是否生成backbone等, 不需要使用的话在yaml中注释掉对应的cfg
        # if 'backbone_cfg' in model_cfg.keys():
        # if 'cost_processor_cfg' in model_cfg.keys():
        # if 'disp_processor_cfg' in model_cfg.keys():

    def forward(self, inputs):
        # input的类型是词典
        # inputs.keys(): dict_keys(['ref_img', 'tgt_img', 'disp_gt', 'mask', 'index'])
        disparity_map, edge = self.net(inputs)

        # print(disparity_map.shape) # 可以去掉这一步，在net里确保返回值的shape为B*H*W
        if disparity_map.dim() == 4 and disparity_map.shape[1] == 1:
            disparity_map = disparity_map.squeeze(1)
        if self.training:
            # 3层词典嵌套
            output = {
                # disparity_map 需要和yaml中loss_cfg/log_prefix参数保持一致, 实质是loss的前缀名
                "training_disp": {
                    "disparity_map": {
                        "disp_ests": [disparity_map],
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"],
                    },
                    # 如果还有多个loss计算，可以传递其他log_prefix
                    # "another loss": {
                    # },
                },
                # visual_summary用于tensorboard可视化
                "visual_summary": {
                    "image/train/image_c": torch.cat([inputs["ref_img"][0], inputs["tgt_img"][0]], dim=1),  # dim=1,上下拼接,shape=C*2H*W
                    "image/train/disp_c": torch.cat([inputs["disp_gt"][0], disparity_map[0]], dim=0),
                    "image/train/edge": edge[0][0], # edge.shape 2*1*H*W
                },
            }
            # max1=disparity_map.max()
            # min1=disparity_map.min()
            # max2=edge.max()
            # min2=edge.min()
            # max3=inputs["disp_gt"][0].max()
            # min3=inputs["disp_gt"][0].min()
        else:
            # val或test的一个batch只取一张图像返回吗
            # test数据集的inputs.keys(): dict_keys(['ref_img', 'tgt_img', 'name', 'index']), 少了disp_gt
            if disparity_map[-1].dim() == 4:  # shape为B*1*H*W, 删除通道为 B*H*W
                disparity_map[-1] = disparity_map[-1].squeeze(1)
            output = {
                "inference_disp": {
                    # disp_est而不是ests, 单数而非复数, 只返回一张图
                    "disp_est": disparity_map[0],
                    # "ref_img" : inputs["ref_img"],
                    # "disp_gt": inputs["disp_gt"],
                    # "mask": inputs["mask"],
                },
                "visual_summary": {
                    "image/train/image_c": torch.cat([inputs["ref_img"][0], inputs["tgt_img"][0]], dim=1),
                    "image/train/disp_c": disparity_map[0],
                    # "image/train/edge": 1,
                },
            }
            # if 'disp_gt' in inputs:
            #     disp_gt = inputs['disp_gt']
            #     output['visual_summary'] = {
            #         'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
            #         'image/val/disp_c': torch.cat([disp_gt[0], disp3[0]], dim=0),
            #     }
        return output


# 未使用, 使用默认的base_trainer
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
        if optimizer_cfg["solver"] == "lamb":
            # use lamb optimizer
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = Lamb(filter(lambda p: p.requires_grad, self.model.parameters()), lr=optimizer_cfg["lr"])
        else:
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = get_attr_from([optim], optimizer_cfg["solver"])
            valid_arg = get_valid_args(optimizer, optimizer_cfg, ["solver"])
            optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer = optimizer

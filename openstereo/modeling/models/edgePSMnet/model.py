import torch
import torch.nn as nn
import time
# import torch.nn.functional as F
# import torch.optim as optim
# from .edgestereo import edgestereo
from modeling.base_model import BaseModel
from base_trainer import BaseTrainer
from utils import get_attr_from, get_valid_args

# from modeling.common.lamb import Lamb  # 优化器, 继承自Optimizer
from .costprocessor import *
from .backbone import *


class edgePSM(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        To use your own trainer, you need to set self.Trainer to your trainer class.
        """
        self.Trainer = edgePSMNetTrainer

    def build_backbone(self, backbone_cfg):
        # """Get the backbone of the model."""
        # if is_dict(backbone_cfg):
        #     Backbone = get_attr_from([backbones], backbone_cfg['type'])
        #     valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
        #     return Backbone(**valid_args)
        # if is_list(backbone_cfg):
        #     Backbone = nn.ModuleList([self.get_backbone(cfg) for cfg in backbone_cfg])
        #     return Backbone
        # raise ValueError("Error type for -Backbone-Cfg-, supported: (A list of) dict.")
        return EdgePSMBackbone()

    # def build_cost_processor(self, cost_processor_cfg):
        # return EdgePSMCostProcessor()
        # return super().build_cost_processor(cost_processor_cfg)

    # def init_parameters(self):
    #     pass

    def forward_step(self, batch_data, device=None):
        T1 = time.clock()
        batch_inputs = self.prepare_inputs(batch_data, device)
        T2 =time.clock()
        print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
        outputs = self.forward(batch_inputs)
        T3 =time.clock()
        print('程序运行时间:%s毫秒' % ((T3 - T2)*1000))
        training_disp, visual_summary = outputs['training_disp'], outputs['visual_summary']
        return training_disp, visual_summary

    # backbone modeling/backbone/PSMNet.py
    # cost processor modeling/cost_processor/PSMNet.py
    # disp processor modeling/disp_processor/PSMNet.py


class edgePSMNetTrainer(BaseTrainer):
    """
    You can define your own trainer class here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "PSMNetTrainer"

import torch
import torch.nn as nn

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

    def init_parameters(self):
        pass

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

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
from .dispprocessor import *

class edgeCoex(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_backbone(self, backbone_cfg):
        valid_args = get_valid_args(EdgeCoexBackbone, backbone_cfg, ['type']) # 有默认值
        return EdgeCoexBackbone(**valid_args)

    def build_cost_processor(self, cost_processor_cfg):
        valid_args = get_valid_args(EdgeCoexProcessor, cost_processor_cfg, ['type']) 
        return EdgeCoexProcessor(**valid_args)
    
    def build_disp_processor(self, disp_processor_cfg):
        valid_args = get_valid_args(EdgeCoExDispProcessor, disp_processor_cfg, ['type']) 
        return EdgeCoExDispProcessor(**valid_args)


    # backbone modeling/backbone/CoEx.py
    # cost processor modeling/cost_processor/CoEx.py
    # disp processor modeling/disp_processor/CoEx.py

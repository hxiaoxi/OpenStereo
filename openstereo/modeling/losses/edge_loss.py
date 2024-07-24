import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class Edge_Smooth_Loss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, beta=1):
        super().__init__(loss_term_weight)
        self.beta = beta

    def forward(self, edge_est, disp_est, mask=None):
        assert edge_est.shape == disp_est.shape  # shape=B*H*W
        b, h, w = edge_est.size()

        # 计算视差图在 x 和 y 方向上的梯度
        grad_x = torch.zeros_like(disp_est)
        grad_y = torch.zeros_like(disp_est)
        grad_x[:, :, 1:] = torch.abs(disp_est[:, :, 1:] - disp_est[:, :, :-1])
        grad_y[:, 1:, :] = torch.abs(disp_est[:, 1:, :] - disp_est[:, :-1, :])

        # 计算边缘图在 x 和 y 方向上的梯度
        edge_grad_x = torch.zeros_like(edge_est)
        edge_grad_y = torch.zeros_like(edge_est)
        edge_grad_x[:, :, 1:] = torch.abs(edge_est[:, :, 1:] - edge_est[:, :, :-1])
        edge_grad_y[:, 1:, :] = torch.abs(edge_est[:, 1:, :] - edge_est[:, :-1, :])

        # 计算加权的视差梯度
        weighted_grad_x = grad_x * torch.exp(-self.beta * edge_grad_x)
        weighted_grad_y = grad_y * torch.exp(-self.beta * edge_grad_y)

        # 使用 mask 进行筛选
        if mask is not None:
            weighted_grad_x = weighted_grad_x * mask
            weighted_grad_y = weighted_grad_y * mask

        # 计算损失
        loss = (weighted_grad_x + weighted_grad_y).sum() / mask.sum()
        self.info.update({"loss": loss})  # info是词典类型
        return loss, self.info

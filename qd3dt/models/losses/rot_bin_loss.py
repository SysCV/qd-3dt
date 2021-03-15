import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weighted_loss
from ..registry import LOSSES

l1_loss = weighted_loss(F.l1_loss)
ce_loss = weighted_loss(F.cross_entropy)


@LOSSES.register_module
class RotBinLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target_bin,
                target_res,
                weight=None,
                avg_factor=None):
        """Compute rotation angle L1 loss and direction classification loss

        Args:
            pred (torch.Tensor): (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
                bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]]
            target_bin (torch.Tensor): (B, 2) [bin1_cls, bin2_cls] 
            target_res (torch.Tensor): (B, 2) [bin1_res, bin2_res]
            weight (torch.Tensor, optional): loss weight. Defaults to None.
            avg_factor (torch.Tensor, optional): Avarage factor when computing the mean of losses. Defaults to None.

        Returns:
            torch.Tensor: Sum of the CE loss and L1 loss
        """

        loss_bin1 = self.loss_weight * ce_loss(
            pred[:, 0:2],
            target_bin[:, 0],
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        loss_bin2 = self.loss_weight * ce_loss(
            pred[:, 4:6],
            target_bin[:, 1],
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        loss_res = torch.zeros_like(loss_bin1)
        if target_bin[:, 0].nonzero().shape[0] > 0:
            idx1 = target_bin[:, 0].nonzero()[:, 0]
            valid_output1 = torch.index_select(pred, 0, idx1.long())
            valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
            loss_sin1 = l1_loss(
                valid_output1[:, 2],
                torch.sin(valid_target_res1[:, 0]),
                weight,
                reduction=self.reduction,
                avg_factor=avg_factor)
            loss_cos1 = l1_loss(
                valid_output1[:, 3],
                torch.cos(valid_target_res1[:, 0]),
                weight,
                reduction=self.reduction,
                avg_factor=avg_factor)
            loss_res += loss_sin1 + loss_cos1
        if target_bin[:, 1].nonzero().shape[0] > 0:
            idx2 = target_bin[:, 1].nonzero()[:, 0]
            valid_output2 = torch.index_select(pred, 0, idx2.long())
            valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
            loss_sin2 = l1_loss(
                valid_output2[:, 6],
                torch.sin(valid_target_res2[:, 1]),
                weight,
                reduction=self.reduction,
                avg_factor=avg_factor)
            loss_cos2 = l1_loss(
                valid_output2[:, 7],
                torch.cos(valid_target_res2[:, 1]),
                weight,
                reduction=self.reduction,
                avg_factor=avg_factor)
            loss_res += loss_sin2 + loss_cos2
        return loss_bin1 + loss_bin2 + loss_res
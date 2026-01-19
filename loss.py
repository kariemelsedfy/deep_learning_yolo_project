"""
Implementation of YOLOv1 Loss Function
Modified to support ONLY B = 1
"""

import torch
import torch.nn as nn
from utils import intersection_over_union  


class YoloLoss(nn.Module):
    """
    Calculate the loss for YOLO (v1) model with B = 1
    """

    def __init__(self, S=7, B=1, C=1):
        super(YoloLoss, self).__init__()
        assert B == 1, "This implementation supports only B = 1"

        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        """
        predictions & target shape:
        (N, S, S, C + 5)
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + 5)
        target = target.reshape(-1, self.S, self.S, self.C + 5)

        # Object existence indicator (I_obj)
        exists_box = target[..., self.C:self.C + 1]

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_predictions = exists_box * predictions[..., self.C + 1:self.C + 5]
        box_targets = exists_box * target[..., self.C + 1:self.C + 5]

        # sqrt on width and height (NO in-place ops)
        pred_xy = box_predictions[..., 0:2]
        pred_wh = box_predictions[..., 2:4]
        tgt_xy = box_targets[..., 0:2]
        tgt_wh = box_targets[..., 2:4]

        pred_wh = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)
        tgt_wh = torch.sqrt(tgt_wh + 1e-6)

        box_predictions = torch.cat([pred_xy, pred_wh], dim=-1)
        box_targets = torch.cat([tgt_xy, tgt_wh], dim=-1)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        object_loss = self.mse(
            torch.flatten(exists_box * predictions[..., self.C:self.C + 1]),
            torch.flatten(exists_box * target[..., self.C:self.C + 1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C + 1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C + 1], start_dim=1),
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2),
        )

        # ================== #
        #   TOTAL LOSS       #
        # ================== #

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

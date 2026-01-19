"""
STUDENT FILE: YOLOv1 model definition

Context:
- This file defines the neural network used for object detection.
- The architecture follows YOLOv1 with convolutional layers
  followed by fully connected layers.
- You are not expected to design a new architecture.
- Focus on understanding how YOLO outputs predictions.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------
# Architecture configuration (provided)
# ---------------------------------------------------------
"""
Each tuple represents:
(kernel_size, number_of_filters, stride, padding)

"M" represents max pooling.

Lists represent repeated convolutional blocks.
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# ---------------------------------------------------------
# Convolutional building block (provided)
# ---------------------------------------------------------
class CNNBlock(nn.Module):
    """
    Convolution + BatchNorm + LeakyReLU
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =========================================================
# ======================== STUDENT TODO ===================
# =========================================================
class Yolov1(nn.Module):
    """
    YOLOv1 model.

    High-level structure:
    1. Convolutional feature extractor
    2. Fully connected prediction head
    """

    def __init__(self, in_channels=3, split_size=7, num_boxes=1, num_classes=1):
        super().__init__()

        self.architecture = architecture_config
        self.in_channels = in_channels
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # TODO:
        # Create the convolutional backbone using the architecture config
        self.darknet = None

        # TODO:
        # Create the fully connected layers that output YOLO predictions
        self.fcs = None

    def forward(self, x):
        """
        Forward pass:
        - extract features
        - flatten
        - produce final predictions
        """
        # TODO
        raise NotImplementedError

    def _create_conv_layers(self, architecture):
        """
        Builds the convolutional feature extractor.

        The architecture configuration describes:
        - convolution layers
        - max pooling layers
        - repeated blocks
        """
        # TODO
        raise NotImplementedError

    def _create_fcs(self):
        """
        Builds the fully connected prediction head.

        The output should encode:
        - class probabilities
        - objectness score
        - bounding box coordinates
        for every grid cell.
        """
        # TODO
        raise NotImplementedError

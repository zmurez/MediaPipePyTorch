import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blazebase import BlazeLandmark, BlazeBlock

class BlazeFaceLandmark(BlazeLandmark):
    """The face landmark model from MediaPipe.
    
    """
    def __init__(self):
        super(BlazeFaceLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 192

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(16),

            BlazeBlock(16, 16, 3, act='prelu'),
            BlazeBlock(16, 16, 3, act='prelu'),
            BlazeBlock(16, 32, 3, 2, act='prelu'),

            BlazeBlock(32, 32, 3, act='prelu'),
            BlazeBlock(32, 32, 3, act='prelu'),
            BlazeBlock(32, 64, 3, 2, act='prelu'),

            BlazeBlock(64, 64, 3, act='prelu'),
            BlazeBlock(64, 64, 3, act='prelu'),
            BlazeBlock(64, 128, 3, 2, act='prelu'),

            BlazeBlock(128, 128, 3, act='prelu'),
            BlazeBlock(128, 128, 3, act='prelu'),
            BlazeBlock(128, 128, 3, 2, act='prelu'),

            BlazeBlock(128, 128, 3, act='prelu'),
            BlazeBlock(128, 128, 3, act='prelu'),
        )


        self.backbone2a = nn.Sequential(
            BlazeBlock(128, 128, 3, 2, act='prelu'),
            BlazeBlock(128, 128, 3, act='prelu'),
            BlazeBlock(128, 128, 3, act='prelu'),
            nn.Conv2d(128, 32, 1, padding=0, bias=True),
            nn.PReLU(32),
            BlazeBlock(32, 32, 3, act='prelu'),
            nn.Conv2d(32, 1404, 3, padding=0, bias=True)
        )

        self.backbone2b = nn.Sequential(
            BlazeBlock(128, 128, 3, 2, act='prelu'),
            nn.Conv2d(128, 32, 1, padding=0, bias=True),
            nn.PReLU(32),
            BlazeBlock(32, 32, 3, act='prelu'),
            nn.Conv2d(32, 1, 3, padding=0, bias=True)
        )

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,)), torch.zeros((0, 468, 3))
            
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        landmarks = self.backbone2a(x).view(-1, 468, 3) / 192
        flag = self.backbone2b(x).sigmoid().view(-1)

        return flag, landmarks
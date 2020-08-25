import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blazebase import BlazeDetector, BlazeBlock


class BlazePalm(BlazeDetector):
    """The palm detection model from MediaPipe. """
    def __init__(self):
        super(BlazePalm, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/hand_tracking/subgraphs/hand_detection_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 2944
        self.num_coords = 18
        self.score_clipping_thresh = 100.0
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0
        self.min_score_thresh = 0.5
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 7

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        self.detection2roi_method = 'box'
        # mediapipe/graphs/hand_tracking/subgraphs/hand_detection_cpu.pbtxt
        self.kp1 = 0
        self.kp2 = 2
        self.theta0 = np.pi/2
        self.dscale = 2.6
        self.dy = -0.5

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            
            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),

            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),

        )
        
        self.backbone2 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.blaze1 = BlazeBlock(256, 256)

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.blaze2 = BlazeBlock(128, 128)

        self.classifier_32 = nn.Conv2d(128, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(256, 2, 1, bias=True)
        self.classifier_8 = nn.Conv2d(256, 6, 1, bias=True)
        
        self.regressor_32 = nn.Conv2d(128, 36, 1, bias=True)
        self.regressor_16 = nn.Conv2d(256, 36, 1, bias=True)
        self.regressor_8 = nn.Conv2d(256, 108, 1, bias=True)
        
    def forward(self, x):
        b = x.shape[0]      # batch size, needed for reshaping later

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)           # (b, 128, 32, 32)        
        y = self.backbone2(x)           # (b, 256, 16, 16)
        z = self.backbone3(y)           # (b, 256, 8, 8)

        y = y + F.relu(self.conv_transpose1(z), True)
        y = self.blaze1(y)

        x = x + F.relu(self.conv_transpose2(y), True)
        x = self.blaze2(x)


        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8(z)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(y)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c3 = self.classifier_32(x)      # (b, 6, 8, 8)
        c3 = c3.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c3 = c3.reshape(b, -1, 1)       # (b, 384, 1)

        c = torch.cat((c3, c2, c1), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(z)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 18)      # (b, 512, 16)

        r2 = self.regressor_16(y)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 18)      # (b, 384, 16)

        r3 = self.regressor_32(x)       # (b, 96, 8, 8)
        r3 = r3.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r3 = r3.reshape(b, -1, 18)      # (b, 384, 16)

        r = torch.cat((r3, r2, r1), dim=1)  # (b, 896, 16)

        return [r, c]

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from lib.models.fusion import Fusion
from lib.models.centernet3d import CenterNet3D

from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.losses.head_distill_loss import compute_head_distill_loss
from lib.losses.feature_distill_loss import compute_backbone_l1_loss
from lib.losses.feature_distill_loss import compute_backbone_resize_affinity_loss
from lib.losses.feature_distill_loss import compute_backbone_local_affinity_loss



class CrossDistillSeparate(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4, flag='training', model_type='distill'):
        print("CrossDistillSeparate!!!!!")
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.centernet_rgb = CenterNet3D(backbone=backbone, neck=neck, num_class=num_class, downsample=downsample, flag=flag, model_type=model_type, modality='rgb')
        self.centernet_depth = CenterNet3D(backbone=backbone, neck=neck, num_class=num_class, downsample=downsample, flag=flag, model_type=model_type, modality='depth')

        for i in self.centernet_depth.parameters():
            i.requires_grad = False


        self.flag = flag


    def forward(self, input, target=None):
        if self.flag == 'testing':
            rgb = input['rgb']
            depth = input['depth']
            
            
            # coin toss to choose rgb or depth
            if np.random.rand() > 5:
                # print("rgb")
                rgb_feat, rgb_outputs, rgb_ = self.centernet_rgb(rgb)
                return rgb_feat, rgb_outputs, rgb_
            else:
                # print("depth")
                depth_feat,  depth_outputs, depth_ = self.centernet_depth(depth)
                return depth_feat, depth_outputs, depth_
            

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    import torch
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "1"
    # net = MonoDistill(backbone='dla34')
    # print(net)

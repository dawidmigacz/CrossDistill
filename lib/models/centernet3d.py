import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from lib.backbones import dla
from lib.backbones.dlaup import DLAUp
from lib.backbones.hourglass import get_large_hourglass_net
from lib.backbones.hourglass import load_pretrian_model

from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections


class CenterNet3D(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4, flag='training', model_type='centernet3d', modality='rgb', drop_prob=0.202):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()
        self.flag = flag
        self.model_type = model_type
        self.modality = modality
        self.feature_flag = backbone
        self.first_level = int(np.log2(downsample))
        self.drop_prob=drop_prob

        self.backbone = getattr(dla, backbone)(pretrained=True, return_levels=True, drop_prob=self.drop_prob)
        channels = self.backbone.channels  # channels list for feature maps generated by backbone
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.neck = DLAUp(channels[self.first_level:], scales_list=scales)   # feature fusion [such as DLAup, FPN]

        # initialize the head of pipeline, according to heads setting.
        self.heads = {'heatmap': num_class, 'offset_2d': 2, 'size_2d' :2, 'depth': 2, 'offset_3d': 2, 'size_3d':3, 'heading': 24}
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(channels[self.first_level], 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            # initialization
            if 'heatmap' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def forward(self, input):
        if self.model_type == 'centernet3d' or self.model_type == 'distill_separate':
            # print(" inference centernet3d, modality: ", self.modality)
            try:
                input = input[self.modality] 
            except:
                pass

        feat_backbone = self.backbone(input)
        feat = self.neck(feat_backbone[self.first_level:])  # first_level = 2

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)

        return feat_backbone, ret, feat


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
    net = CenterNet3D(backbone='dla34')
    print(net)

    # # pretrained_dict = torch.load('experiments/example/rgb_pretrain.pth')
    # # net.load_state_dict(pretrained_dict)

    # from lib.helpers.save_helper import load_checkpoint
    # load_checkpoint(model=net,
    #                 optimizer=None,
    #                 filename='experiments/example/rgb_pretrain.pth',
    #                 map_location="cuda:0",
    #                 logger=None)


    
    # import torchvision

  
    # resizer = torchvision.transforms.Resize([384, 1280])
    # input = resizer(torchvision.io.read_image("data/KITTI/object/training/image_2/000039.png")).resize(1, 3, 384, 1280).float()

    # print(input.shape, input.dtype)
    # output = net(input)
    # print(len(output[0]))
    # for i in range(len(output[0])):
    #     print(f"Size of element {i}: {output[0][i].size()}")

    # print("o2 ", output[2].size())
    # # print("output ", output[1].keys())
    # for i in output[1].keys():
    #     print(i, output[1][i].size())
    # #     img = output[1][i].detach().cpu().numpy()[0, 0]
    # #     img = (img * 255).astype(np.uint8)
    # #     cv2.imshow(i, img)
    # #     cv2.waitKey(0)
    # #     cv2.destroyAllWindows()

    # dets = extract_dets_from_outputs(output[1], K=100)
    # print(dets.size())
    # print(dets)
    # import matplotlib.pyplot as plt

    # # img = dets.detach().cpu().numpy()[0]
    # # plt.imshow(img)
    # # plt.show()

    # cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
    #                         [1.52563191, 1.62856739, 3.52588311],
    #                         [1.73698127, 0.59706367, 1.76282397]], dtype=np.float32) 

    # resolution = np.array([1280, 384])
    # img_size = np.array(input.size())
    # print(img_size)
    # features_size = resolution // 4
    # print(features_size)
    # info = {'img_id': [170],
    #             'img_size': [[384, 1280]],
    #             'bbox_downsample_ratio': [[4.0, 4.0]]}

    # from lib.datasets.kitti.kitti_utils import get_calib_from_file
    # # import Calibration 
    # from lib.datasets.kitti.kitti_utils import Calibration

    # calib_path = 'data/KITTI/object/training/calib/000039.txt'
    # calibs = [Calibration(calib_path)]

    # detects = decode_detections(dets=dets.detach().numpy(),
    #                                  info=info,
    #                                  calibs=calibs,
    #                                  cls_mean_size=cls_mean_size,
    #                                  threshold=0.2)
    # # print(output[1]['heading'].size())
    # print(detects)

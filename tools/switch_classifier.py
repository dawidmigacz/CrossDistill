import os
import sys
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

import random
import yaml
import argparse
import datetime
import os

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.uncertainty_helper import Uncertainter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# parser = argparse.ArgumentParser(description='Monocular 3D Object Detection')
# parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
# parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
# parser.add_argument('-u', '--uncertainty_only', action='store_true', default=False, help='uncertainty only')

# args = parser.parse_args()

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)


import os
import shutil



dir_path = './depth_outputs/data'
dir_path2 = './depth_outputs/folders_val'
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        new_dir_path = os.path.join(dir_path2, filename[:-4])
        os.makedirs(new_dir_path, exist_ok=True)
        shutil.move(os.path.join(dir_path, filename), os.path.join(new_dir_path, filename))



# assert (os.path.exists(args.config))
# cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
# print(cfg)






# log_path = ROOT_DIR + "/experiments/example/logs/"
# if os.path.exists(log_path):
#     pass
# else:
#     os.mkdir(log_path)
# log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# logger = create_logger(log_path, log_file)


# dataset_cfg = {'type': 'KITTI',
#                 'batch_size': 1,
#                 'use_3d_center': True,
#                 'class_merging': False, 
#                 'use_dontcare': False, 
#                 'bbox2d_type': 'anno', 
#                 'meanshape': False, 
#                 'writelist': ['Car'], 
#                 'random_flip': 0.5, 
#                 'random_crop': 0.5, 
#                 'scale': 0.4, 
#                 'shift': 0.0, 
#                 'uncertainty': False
#                 }


# tester_cfg = {'type': 'KITTI', 
#               'mode': 'single', 
#               'checkpoint': './models/rgb_pretrain.pth', 
#               'checkpoints_dir': 'distill', 
#               'threshold': 0.2, 
#               'bayes_n': None, 
#               'model_type': 'centernet3d', 
#               'uncertainty_threshold': -0.1
#               }

# dir_path = './depth_outputs/folders_val'
# for foldername in os.listdir(dir_path):
#     if os.path.isdir(os.path.join(dir_path, foldername)):
#         with open('../../data/KITTI/ImageSets/val.txt', 'w') as f:
#             f.write(f'{foldername}\n')
#     train_loader, test_loader  = build_dataloader(dataset_cfg)
#     print(dir_path+foldername)
#     res=test_loader.dataset.eval(results_dir=dir_path+foldername, logger=logger)
#     print(res)
#     print(foldername)
#     raise KeyboardInterrupt



# wandb.init(
# project=cfg['wandb']['project'],
# notes=cfg['wandb']['notes'],
# tags=cfg['wandb']['tags'],
# config = cfg
# )








# class SwitchClassifier(nn.Module):
#     def __init__(self):
#         super(SwitchClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.fc = nn.Linear(32 * 617 * 228, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# model = SwitchClassifier()

# print(model)




# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Assume we have train_loader and test_loader for training and testing data

# # Training the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# # Evaluating the model
# model.eval() 
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         outputs = model(images)
#         predicted = torch.round(torch.sigmoid(outputs))
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f'Accuracy of the model on test images: {100 * correct / total}%')
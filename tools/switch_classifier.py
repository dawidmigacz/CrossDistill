import os
import sys
from tqdm import tqdm
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


dataset_cfg = {'type': 'KITTI',
                'batch_size': 8,
                'use_3d_center': True,
                'class_merging': False, 
                'use_dontcare': False, 
                'bbox2d_type': 'anno', 
                'meanshape': False, 
                'writelist': ['Car'], 
                'random_flip': 0.5, 
                'random_crop': 0.5, 
                'scale': 0.4, 
                'shift': 0.0, 
                'uncertainty': False,
                'values_flag': True,
                }



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwitchClassifier(nn.Module):
    def __init__(self):
        super(SwitchClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(983040, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SwitchClassifier().to(device)

print(model)

lr = 0.00003
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Assume we have train_loader and test_loader for training and testing data

wandb.init(
    project="switch-classifier",
    config={
        "learning_rate": lr,
        "architecture": "CNN",
        "dataset": "KITTI"
    }
)
train_loader, test_loader = build_dataloader(dataset_cfg)


# # Training the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     for batch_idx, (inputs, targets, info) in tqdm(enumerate(train_loader)):
#         optimizer.zero_grad()
#         outputs = model(inputs['rgb'])
#         # print(outputs)
#         # print(targets['values_depth']-targets['values_rgb'])
#         loss = criterion(outputs, (targets['values_depth']-targets['values_rgb']).unsqueeze(1))
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

num_epochs = 0
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets, info) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Move the inputs and targets to the GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        outputs = model(inputs['rgb'])
        
        loss = criterion(outputs, (targets['values_depth']-targets['values_rgb']).unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
# Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"checkpoint_{epoch+1}.pth")

wandb.finish()
# Evaluating the model

# load saved models from checkpoints
model = SwitchClassifier().to(device)
for epoch in range(80, 0, -10):
    checkpoint = torch.load(f"checkpoint_{epoch}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochh = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded model from epoch {epochh}, loss: {loss.item()}")

    model.eval()
    #calculate test MSE loss
    test_loss = 0
    with torch.no_grad():
        for inputs, targets, info in tqdm(test_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(inputs['rgb'])
            test_loss += criterion(outputs, (targets['values_depth']-targets['values_rgb']).unsqueeze(1).float()).item()
    test_loss /= len(test_loader)
    print(f"Test MSE loss: {test_loss}")











# tester_cfg = {'type': 'KITTI', 
#               'mode': 'single', 
#               'checkpoint': './models/rgb_pretrain.pth', 
#               'checkpoints_dir': 'distill', 
#               'threshold': 0.2, 
#               'bayes_n': None, 
#               'model_type': 'centernet3d', 
#               'uncertainty_threshold': -0.1
#               }
# n = 0


# class OverlapLogger:
#     def __init__(self, filename):
#         self.filename = filename
#         self.current_image = None
#         #clear file
#         with open(self.filename, 'w') as file:
#             file.write('')

#     def log(self, message):
#         with open(self.filename, 'a') as file:
#             file.write(f'{self.current_image} {message}\n')

# # Usage
# overlap_logger = OverlapLogger('logfile_depth.txt')

# dir_path = './depth_outputs/folders_val'
# os.makedirs(dir_path, exist_ok=True)
# for foldername in os.listdir(dir_path):
#     if os.path.isdir(os.path.join(dir_path, foldername)):
#         with open('../../data/KITTI/ImageSets/val.txt', 'w') as f:
#             f.write(f'{foldername}\n')
#     train_loader, test_loader  = build_dataloader(dataset_cfg)
#     print(dir_path+foldername)
#     overlap_logger.current_image = foldername
#     res=test_loader.dataset.eval(results_dir=dir_path+'/'+foldername, logger=logger, overlap_logger=overlap_logger)

#     logger.info(foldername)
#     logger.info(res)
#     print(foldername)
    # n+=1
    # if n==5:
    #     raise Exception('stop')



# wandb.init(
# project=cfg['wandb']['project'],
# notes=cfg['wandb']['notes'],
# tags=cfg['wandb']['tags'],
# config = cfg
# )














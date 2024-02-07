from tqdm import tqdm
import torch
import torch.nn as nn
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helpers.dataloader_helper import build_dataloader


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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SwitchClassifier().to(device)

checkpoint = torch.load("checkpoint_30.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
dataset_cfg = {'type': 'KITTI',
                'batch_size': 1,
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

train_loader, test_loader = build_dataloader(dataset_cfg)


test_loss = 0
with torch.no_grad():
    for inputs, targets, info in tqdm(test_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(inputs['rgb'])
        print(info["img_id"].item(), outputs.item())
        # append the above to file
        with open('switches.txt', 'a') as f:
            # f.write(f"{info['img_id'].item()} {outputs.item()}\n")
            f.write(f"{str(info['img_id'].item()).zfill(6)} {outputs.item()}\n")
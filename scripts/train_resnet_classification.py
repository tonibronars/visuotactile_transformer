#!/usr/bin/env python

import torch
import timm
import glob
import numpy as np
import cv2
import os
import shutil
import sys
import importlib
import time
import datetime
import os
from torch.autograd import Variable

from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from visuotactile_transformer.classes.dataset_classification import VisuotactileDataset

vision_only = False
tactile_only = False
saving_folder = 'visuotactile_resnet_classification_erase_big'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
save_path = '/home/gridsan/bronars/src/visuotactile_transformer/models/{}/'.format(saving_folder)
os.makedirs(save_path,exist_ok=True)

os.chdir('/home/gridsan/bronars/src/visuotactile_transformer/')
os.system('cp scripts/train_resnet_classification.py models/{}/train_resnet_classification.py'.format(saving_folder))
os.system('cp classes/dataset_classification.py models/{}/dataset_classification.py'.format(saving_folder))
os.chdir('/home/gridsan/bronars/src/')

def save_checkpoint(state, filename):
    torch.save(state, filename)

class VisionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50")
        self._fc = torch.nn.Linear(1000,4)
    def forward(self, x):
        x = self.model(x)
        x = self._fc(x)
        return x
class TactileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50")
        self._fc = torch.nn.Linear(1000,4)
    def forward(self, x):
        batch_size ,_,_,_ =x.shape
        x = self.model(x)
        x = self._fc(x)
        return x
class VisuotactileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_model = timm.create_model("resnet50")
        self.tac_model = timm.create_model("resnet50")
        self._fc = torch.nn.Linear(2000,4)
    def forward(self, vis, tac):
        x1 = self.vis_model(vis)
        x2 = self.tac_model(tac)
        x = torch.cat((x1, x2), 1)
        x = self._fc(x)
        return x



# Define training parameters
def main():
    from argparse import Namespace
    args = Namespace(input_pose=True, is_binary=False, is_vision=False, batch_size=64,  lr=0.0001, momentum=0.9, weight_decay=0)

    if vision_only: model = VisionNet()
    elif tactile_only: model = TactileNet()
    else: model = VisuotactileNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if torch.cuda.is_available():
        model.cuda()

    # Create training datasets
    train_ds = VisuotactileDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)

    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    lowest_loss = 1000
    for epoch in range(51):        
      for step, item in enumerate(train_loader):
        # Change input array into list with each batch being one element
        tactile = item['tactile']
        vision = item['vision']
        target = item['target']

        # Send to GPU if available
        tactile, vision, target = tactile.to(device), vision.to(device), target.to(device)
        b_tactile = Variable(tactile)   # batch x (image)
        b_vision = Variable(vision)   # batch y (target)

        # Feed through model
        if vision_only: y_pred = model(b_vision)
        elif tactile_only: y_pred = model(b_tactile)
        else: y_pred = model(b_vision, b_tactile)

        # Compute the loss
        logits = torch.nn.Softmax(dim=-1)(y_pred).to(device)
        loss = criterion(logits, target)

        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()

        if step % 5 == 0:
          print('Epoch: ', epoch, '| train loss: ', loss)
      scheduler.step()
      if loss.item() < lowest_loss:
        save_checkpoint(model, save_path + 'visuotactile_model_best.pth')
        lowest_loss = loss.item()
      if epoch % 10 == 0:
        save_checkpoint(model, save_path + 'visuotactile_model_{:04d}.pth'.format(epoch))


if __name__ == '__main__':
    print('train module is being executed')
    main()

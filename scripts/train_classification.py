#!/usr/bin/env python

import torch
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
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTForImageClassification, ViTModel
from visuotactile_transformer.classes.dataset_classification import VisuotactileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def save_checkpoint(state, filename):
    torch.save(state, filename)

tactile_only = False
vision_only = True
saving_folder = 'vision_transformer_classification_erase_big2'


save_path = '/home/gridsan/bronars/src/visuotactile_transformer/models/{}/'.format(saving_folder)
os.makedirs(save_path,exist_ok=True)

os.chdir('/home/gridsan/bronars/src/visuotactile_transformer/')
os.system('cp scripts/train_resnet_classification.py models/{}/train_resnet_classification.py'.format(saving_folder))
os.system('cp classes/dataset_classification.py models/{}/dataset_classification.py'.format(saving_folder))
os.chdir('/home/gridsan/bronars/src/')

feature_extractor = ViTFeatureExtractor.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models") #'google/vit-base-patch16-224-in21k')

class VisionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models")
        self._fc = torch.nn.Linear(197*768,4)
    def forward(self, x):
        x, loss = self.model(x, None).to_tuple()
        x = x.flatten(start_dim=1)
        x = self._fc(x)
        return x
class TactileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models")
        self._fc = torch.nn.Linear(197*768,4)
    def forward(self, x):
        x, loss = self.model(x, None).to_tuple()
        x = x.flatten(start_dim=1)
        x = self._fc(x)
        return x
class VisuotactileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_model = ViTModel.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models")
        self.tac_model = ViTModel.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models")
        self._fc = torch.nn.Linear(2*197*768,4)
    def forward(self, vis, tac):
        x1, loss = self.vis_model(vis, None).to_tuple()
        x2, loss = self.tac_model(tac, None).to_tuple()
        x1 = x1.flatten(start_dim=1)
        x2 = x2.flatten(start_dim=1)
        x = torch.cat((x1, x2), 1)
        x = self._fc(x)
        return x

# Define training parameters
def main():
    from argparse import Namespace
    args = Namespace(input_pose=True, is_binary=False, is_vision=False, batch_size=64,  lr=0.00001, momentum=0.9, weight_decay=0)

    # Initialize vision and tactile transformers
    print('initializing pretrained...')
    if vision_only: model = VisionNet()
    elif tactile_only: model = TactileNet()
    else: model = VisuotactileNet()

    print('Pretrained initialized!')
    if torch.cuda.is_available():
        model.cuda()

    # Create training datasets
    train_ds = VisuotactileDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)

    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    lowest_loss = 1000000
    for epoch in range(51):        
      for step, item in enumerate(train_loader):
        tactile = item['tactile']
        vision = item['vision']
        target = item['target']
        tactile = np.split(np.squeeze(np.array(tactile)), args.batch_size)
        vision = np.split(np.squeeze(np.array(vision)), args.batch_size)

        # Remove unecessary dimension
        for index, array in enumerate(tactile):
          tactile[index] = np.squeeze(array)
        for index, array in enumerate(vision):
          vision[index] = np.squeeze(array)

        '''
        print('before feature extractor..')
        print('vision')
        print(np.max(vision[0]))
        print(np.min(vision[0]))
        print('tactile')
        print(np.max(tactile[0]))
        print(np.min(tactile[0]))
        '''

        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        tactile = torch.tensor(np.stack(feature_extractor(tactile)['pixel_values'], axis=0))
        vision = torch.tensor(np.stack(feature_extractor(vision)['pixel_values'], axis=0))

        '''
        print('after feature extractor..')
        print('vision')
        print(torch.max(vision))
        print(torch.min(vision))
        print('tactile')
        print(torch.max(tactile))
        print(torch.min(tactile))
        '''

        # Send to GPU if available
        tactile, vision, target  = tactile.to(device), vision.to(device), target.to(device)
        b_tactile = Variable(tactile)   # batch x (image)
        b_vision = Variable(vision)   # batch y (target)

        # Feed through model
        if tactile_only: y_pred = model(b_tactile)
        elif vision_only: y_pred = model(b_vision)
        else: y_pred = model(Variable(vision), Variable(tactile))
 
        #print(y_pred)
        logits = torch.nn.Softmax(dim=-1)(y_pred).to(device)
        #print(logits)

        loss = criterion(logits, target)

        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()

        if step % 5 == 0:
          print('Epoch: ', epoch, '| train loss: ', loss)

      scheduler.step()
      if loss.item() < lowest_loss:
        save_checkpoint(model, save_path + 'visuotactile_model_best.bin')
        lowest_loss = loss.item()
      if epoch % 10 == 0:
        save_checkpoint(model, save_path + 'visuotactile_model_{:04d}.bin'.format(epoch))

if __name__ == '__main__':
    print('train module is being executed')
    main()

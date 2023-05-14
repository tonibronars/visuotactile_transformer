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
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
from visuotactile_transformer.classes.dataset import VisuotactileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
save_path = '/home/gridsan/bronars/src/visuotactile_transformer/models/resnet/'
os.makedirs(save_path,exist_ok=True)

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Define training parameters
def main():
    from argparse import Namespace
    args = Namespace(input_pose=True, is_binary=False, is_vision=False, batch_size=16,  lr=0.00001, momentum=0.9, weight_decay=0)

    vision_model = timm.create_model("resnet50")
    for p in vision_model.parameters():
        p.requires_grad = True
    tactile_model = timm.create_model("resnet50")
    for p in tactile_model.parameters():
        p.requires_grad = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if torch.cuda.is_available():
        vision_model.cuda()
        tactile_model.cuda() 

    # Create training datasets
    train_ds = VisuotactileDataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)

    # Create the optimizer
    #optimizer = torch.optim.SGD(tactile_model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    params = [{"params": vision_model.parameters(), "lr": args.lr},
                {"params": tactile_model.parameters(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(params, args.lr,
                                weight_decay=args.weight_decay)
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(10001):        
      for step, item in enumerate(train_loader):
        # Change input array into list with each batch being one element
        tactile = item['tactile']
        vision = item['vision']

        # Send to GPU if available
        tactile, vision  = tactile.to(device), vision.to(device)
        b_tactile = Variable(tactile)   # batch x (image)
        b_vision = Variable(vision)   # batch y (target)

        # Feed through model
        tactile_output = tactile_model(b_tactile)
        vision_output = vision_model(b_vision)

        # Compute the loss
        logits = torch.nn.Softmax(dim=-1)(tactile_output.reshape(args.batch_size, -1) @ vision_output.reshape(args.batch_size, -1).T)
        loss = criterion(logits, torch.arange(args.batch_size).to(device))

        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()

        if step % 5 == 0:
          print('Epoch: ', epoch, '| train loss: ', loss)
      if epoch % 10 == 0:
        save_checkpoint(tactile_model, save_path + 'tactile_model_{:04d}.bin'.format(epoch))
        save_checkpoint(vision_model, save_path + 'vision_model_{:04d}.bin'.format(epoch))

if __name__ == '__main__':
    print('train module is being executed')
    main()

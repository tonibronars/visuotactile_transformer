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
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
from visuotactile_transformer.classes.dataset import VisuotactileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Define training parameters
def main():
    from argparse import Namespace
    args = Namespace(input_pose=True, is_binary=False, is_vision=False, batch_size=16,  lr=0.00001, momentum=0.9, weight_decay=0)

    # Initialize vision and tactile transformers
    print('initializing pretrained...')
    feature_extractor = ViTFeatureExtractor.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models") #'google/vit-base-patch16-224-in21k')
    vision_model = ViTModel.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models") #('google/vit-base-patch16-224-in21k')
    tactile_model = ViTModel.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models") #('google/vit-base-patch16-224-in21k')
    print('initializing pretrained is complete!')

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
        tactile = np.split(np.squeeze(np.array(tactile)), args.batch_size)
        vision = np.split(np.squeeze(np.array(vision)), args.batch_size)

        # Remove unecessary dimension
        for index, array in enumerate(tactile):
          tactile[index] = np.squeeze(array)
        for index, array in enumerate(vision):
          vision[index] = np.squeeze(array)

        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        tactile = torch.tensor(np.stack(feature_extractor(tactile)['pixel_values'], axis=0))
        vision = torch.tensor(np.stack(feature_extractor(vision)['pixel_values'], axis=0))

        # Send to GPU if available
        tactile, vision  = tactile.to(device), vision.to(device)
        b_tactile = Variable(tactile)   # batch x (image)
        b_vision = Variable(vision)   # batch y (target)

        # Feed through model
        tactile_output, loss = tactile_model(b_tactile, None).to_tuple()
        vision_output, loss = vision_model(b_vision, None).to_tuple()

        # Compute the loss
        logits = torch.nn.Softmax(dim=-1)(tactile_output.reshape(args.batch_size, -1) @ vision_output.reshape(args.batch_size, -1).T)
        loss = criterion(logits, torch.arange(args.batch_size).to(device))

        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()

        if step % 5 == 0:
          print('Epoch: ', epoch, '| train loss: ', loss)
      if epoch % 10 == 0:
        save_checkpoint(tactile_model,'/home/gridsan/bronars/src/visuotactile_transformer/models/checkpoints/tactile_model_{:04d}.bin'.format(epoch))
        save_checkpoint(vision_model,'/home/gridsan/bronars/src/visuotactile_transformer/models/checkpoints/vision_model_{:04d}.bin'.format(epoch))

if __name__ == '__main__':
    print('train module is being executed')
    main()

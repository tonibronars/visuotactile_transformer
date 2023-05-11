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
from torch.autograd import Variable

from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
from classes.dataset import VisuotactileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Define training parameters
from argparse import Namespace
args = Namespace(input_pose=True, is_binary=False, is_vision=False, batch_size=2,  lr=0.5, momentum=0.9, weight_decay=0)


# Initialize vision and tactile transformers
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
tactile_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

if torch.cuda.is_available():
    vision_model.cuda()
    tactile_model.cuda()

# Create training datasets
train_ds = VisuotactileDataset()
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)

# Create the optimizer
optimizer = torch.optim.SGD(tactile_model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(5):        
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

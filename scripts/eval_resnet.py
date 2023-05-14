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
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from visuotactile_transformer.classes.dataset import VisuotactileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device: ', device)

# Load checkpoint
tactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/resnet/tactile_model_0050.bin',map_location='cpu')
tactile_model.eval()
vision_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/resnset/vision_model_0050.bin',map_location='cpu')
vision_model.eval()

if torch.cuda.is_available():
    vision_model.cuda()
    tactile_model.cuda()

# Evaluate on training data
data_path = os.environ['HOME'] + '/src/visuotactile_transformer/data/'
data_folders = ['pin_big_subset/', 'aux_big/', 'stud_big/', 'usb_big/']
list_shapes = []
classes = []
for i,df in enumerate(data_folders):
    list_shapes.extend(glob.glob(data_path + df + 'local_shape*1.png'))
    classes.append(i)

import random
c = list(zip(list_shapes, classes))
random.shuffle(c)
list_shapes, classes = zip(*c)

#list_shapes = np.random.permutation(list_shapes)

transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),])

def process_img(path, transforms, model, device, vision=False):
    # load img and transfrom to ViT size
    with torch.no_grad():
        if vision: img = Image.fromarray(np.uint8(np.load(path)*255))
        else: img = Image.open(path)
        img = transforms(img)
        img = torch.tensor(img).float()
        # pass through model
        feat = img.to(device)
        feat = Variable(feat)
        output  = model(feat)
        return output


all_tactile = []
all_vision = []
for i, ls in enumerate(list_shapes[:100]):
    print('iteration ', i)
    tactile = process_img(ls, transforms, tactile_model, device)
    vision = process_img(ls.replace('local_shape','depth').replace('png','npy'), transforms, vision_model, device, vision=True)
    all_tactile.append(tactile.reshape(1,-1))
    all_vision.append(vision.reshape(1,-1))

from sklearn.manifold import TSNE
tsne_tactile = TSNE(n_components=3).fit_transform(torch.cat(all_tactile,dim=0).cpu().detach().numpy())
tsne_vision = TSNE(n_components=3).fit_transform(torch.cat(all_vision,dim=0).cpu().detach().numpy())

import matplotlib.pyplot as plt
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(tsne_tactile[:,0],tsne_tactile[:,1], tsne_tactile[:,2], c=classes[:100],cmap='tab10')
ax.scatter(tsne_vision[:,0],tsne_vision[:,1], tsne_vision[:,2], c=classes[:100], cmap='tab10')plt.show()
import pdb; pdb.set_trace()

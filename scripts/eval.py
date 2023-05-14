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
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
from visuotactile_transformer.classes.dataset import VisuotactileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device: ', device)

# Load checkpoint
tactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/checkpoints/save/tactile_model_0300.bin',map_location='cpu')
tactile_model.eval()
vision_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/checkpoints/save/vision_model_0300.bin',map_location='cpu')
vision_model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models")

if torch.cuda.is_available():
    vision_model.cuda()
    tactile_model.cuda()

# Evaluate on training data
list_shapes = glob.glob(os.environ['HOME'] + '/src/visuotactile_transformer/data/pin_big_subset/local_shape*.png')

transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),])

def process_img(path, transforms, feature_extractor, model, device, vision=False):
    # load img and transfrom to ViT size
    with torch.no_grad():
        if vision: img = Image.fromarray(np.uint8(np.load(path)*255))
        else: img = Image.open(path)
        img = transforms(img)
        img = torch.tensor(img).float()
        # extract ViT features
        feat = np.split(np.squeeze(np.array(img)), 1)
        for index, array in enumerate(feat):
            feat[index] = np.squeeze(array)
        feat = torch.tensor(np.stack(feature_extractor(feat)['pixel_values'], axis=0))
        # pass through model
        feat = feat.to(device)
        feat = Variable(feat)
        output, loss  = model(feat, None).to_tuple()
        del loss
        return output


all_tactile = []
all_vision = []
for i, ls in enumerate(list_shapes):
    print('iteration ', i)
    tactile = process_img(ls, transforms, feature_extractor, tactile_model, device)
    vision = process_img(ls.replace('local_shape','depth').replace('png','npy'), transforms, feature_extractor, vision_model, device, vision=True)
    all_tactile.append(tactile.reshape(1,-1))
    all_vision.append(vision.reshape(1,-1))

from sklearn.manifold import TSNE
tsne_tactile = TSNE(n_components=3).fit_transform(torch.cat(all_tactile,dim=0).cpu().detach().numpy())
tsne_vision = TSNE(n_components=3).fit_transform(torch.cat(all_vision,dim=0).cpu().detach().numpy())

import matplotlib.pyplot as plt
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(tsne_tactile[:,0],tsne_tactile[:,1], tsne_tactile[:,2])
ax.scatter(tsne_vision[:,0],tsne_vision[:,1], tsne_vision[:,2])
plt.show()
import pdb; pdb.set_trace()

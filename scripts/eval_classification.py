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
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTForImageClassification, ViTModel
from visuotactile_transformer.classes.dataset_classification import VisuotactileDataset, Normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device: ', device)

tactile_only = False
vision_only = False
model_name = 'visuotactile_transformer_classification_ang' 

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

# Load checkpoint
'''
print('loadking check')
if tactile_only: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/classification_tactile_transformer/visuotactile_model_best.bin',map_location='cpu')
elif vision_only: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/classification_vision_transformer/visuotactile_model_0500.bin',map_location='cpu')
else: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/classification_vision_transformer2/visuotactile_model_best.bin',map_location='cpu') ## DID NOT FINISH 500 EPOCHS
'''
visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/{}/visuotactile_model_best.bin'.format(model_name),map_location='cpu')
visuotactile_model.eval()

visuotactile_model.eval()
print('done loadking check')

if torch.cuda.is_available():
    visuotactile_model.cuda()

# Evaluate on training data
data_path = os.environ['HOME'] + '/src/visuotactile_transformer/data/'
data_folders = ['pin_big_subset_test/', 'aux_big_test/', 'stud_big_test/', 'usb_big_test/']
list_shapes = []
true_class = []
for i,df in enumerate(data_folders):
    list_shapes.extend(glob.glob(data_path + df + 'local_shape*1.png'))
    true_class.extend([i] * len(glob.glob(data_path + df + 'local_shape*1.png')))

import random
c = list(zip(list_shapes, true_class))
random.shuffle(c)
list_shapes, true_class = zip(*c)

#list_shapes = np.random.permutation(list_shapes)

transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),])

vision_transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),
          Normalize()])

def process_img(path, vision_transforms, transforms, feature_extractor, model, device):
    # load img and transfrom to ViT size
    with torch.no_grad():
        # dataset processing
        vision_img = Image.fromarray(np.uint8(np.load(path.replace('local_shape','depth').replace('png','npy'))*255))
        img = Image.open(path)
        vision_img = vision_transforms(vision_img)
        img = transforms(img)
        vision_img = torch.tensor(vision_img).float()
        img = torch.tensor(img).float()

        # feature processing
        tactile = np.split(np.squeeze(np.array(img)), 1)
        vision = np.split(np.squeeze(np.array(vision_img)), 1)

        # Remove unecessary dimension
        for index, array in enumerate(tactile):
          tactile[index] = np.squeeze(array)
        for index, array in enumerate(vision):
          vision[index] = np.squeeze(array)

        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        tactile = torch.tensor(np.stack(feature_extractor(tactile)['pixel_values'], axis=0))
        vision = torch.tensor(np.stack(feature_extractor(vision)['pixel_values'], axis=0))

        # Send to GPU if available
        tactile, vision = tactile.to(device), vision.to(device)
        b_tactile = Variable(tactile)   # batch x (image)
        b_vision = Variable(vision)   # batch y (target)

        # Feed through model
        if tactile_only: output = model(b_tactile)
        elif vision_only: print('using vision only'); output = model(b_vision)
        else: output = model(Variable(vision), Variable(tactile))

        return output


predicted_class = []
for i, ls in enumerate(list_shapes):
    print(ls)
    output = process_img(ls, vision_transforms, transforms, feature_extractor, visuotactile_model, device)
    predicted_class.append(np.argmax(torch.nn.Softmax()(output)).item())
    print('true class ', true_class[i])
    print('predicted class ', predicted_class[-1])

print('fraction classified correctly: ', len(np.where(np.array(predicted_class) == np.array(true_class))[0])/len(list_shapes))
np.save(os.environ['HOME'] + '/src/visuotactile_transformer/eval/predicted_class_{}.npy'.format(model_name),predicted_class)
np.save(os.environ['HOME'] + '/src/visuotactile_transformer/eval/true_class_{}.npy'.format(model_name),true_class)

import pdb; pdb.set_trace()

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
import mplcursors

from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomRotation, CenterCrop, RandomAffine
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
from visuotactile_transformer.classes.dataset import VisuotactileDataset, Normalize, Morphological

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device: ', device)

type = 'resnet'
number = 1
model_name = '{}_{}'.format(type,number)

# Load checkpoint
tactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/{}/tactile_model_best.bin'.format(model_name),map_location='cpu')
tactile_model.eval()
vision_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/{}/vision_model_best.bin'.format(model_name),map_location='cpu')
vision_model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained(os.environ['HOME'] + "/src/visuotactile_transformer/models")

if torch.cuda.is_available():
    vision_model.cuda()
    tactile_model.cuda()

# Evaluate on training data
data_path = os.environ['HOME'] + '/src/visuotactile_transformer/data/'
data_folders = ['pin_big_subset_test/', 'aux_big_test/', 'stud_big_test/', 'usb_big_test/']
list_shapes = []
classes = []
for i,df in enumerate(data_folders):
    list_shapes.extend(glob.glob(data_path + df + 'local_shape*1.png'))
    classes.extend([i] * len(glob.glob(data_path + df + 'local_shape*1.png')))
import random
#import pdb; pdb.set_trace()
c = list(zip(list_shapes, classes))
random.shuffle(c)
list_shapes, classes = zip(*c)
#import pdb; pdb.set_trace()

#list_shapes = np.random.permutation(list_shapes)

if number == 1:
    tactile_transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),])
    vision_transforms = Compose([
        Grayscale(num_output_channels=3),
        #CenterCrop((random_crop_size, random_crop_size)),
        Resize((224, 224)),
        ToTensor(),
        Normalize(),
        #RandomAffine(degrees=5,translate=(0.5,0.5),fill=1),
])

elif number == 2:
    tactile_transforms = Compose([
          RandomRotation(5, fill=255),
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),
          Morphological()
])
    random_crop_size = np.random.randint(75, 96)
    vision_transforms = Compose([
        Grayscale(num_output_channels=3),
        CenterCrop((random_crop_size, random_crop_size)),
        Resize((224, 224)),
        ToTensor(),
        Normalize(),
        #RandomAffine(degrees=5,translate=(0.5,0.5),fill=1),
])

elif number == 3:
    tactile_transforms = Compose([
          RandomRotation(5, fill=255),
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),
          Morphological()
])
    random_crop_size = np.random.randint(75, 96)
    vision_transforms = Compose([
        Grayscale(num_output_channels=3),
        CenterCrop((random_crop_size, random_crop_size)),
        Resize((224, 224)),
        ToTensor(),
        Normalize(),
        RandomAffine(degrees=5,translate=(0.5,0.5),fill=1),
])

else:
    print('transformations undefined')
    assert(False)

def process_img(path, transforms, feature_extractor, model, device, type, vision=False):
    # load img and transfrom to ViT size
    with torch.no_grad():
        if vision: img = Image.fromarray(np.uint8(np.load(path)*255))
        else: img = Image.open(path)

        if type == 'transformer':
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
        else:
            img = transforms(img)
            if vision == False and number == 2: img = torch.squeeze(torch.tensor(img),dim=0).float()
            else: img = torch.tensor(img).float()
            # pass through model
            feat = img.to(device)
            feat = Variable(feat)
            output  = model(torch.unsqueeze(feat,0))

        return output

import matplotlib.pyplot as plt
def get_every_other_color(cmap_name,light=False):
    cmap = plt.get_cmap(cmap_name)
    colors = []
    if not light:
        for i in range(cmap.N):
            if i % 2 == 0:
                colors.append(cmap(i))
    else:
        for i in range(cmap.N):
            if (i+1) % 2 == 0:
                colors.append(cmap(i))
    return colors

from matplotlib.colors import ListedColormap
def create_custom_colormap(colors):
    new_cmap = ListedColormap(colors)
    return new_cmap

all_tactile = []
all_vision = []
for i, ls in enumerate(list_shapes):
    print('iteration ', i)
    tactile = process_img(ls, tactile_transforms, feature_extractor, tactile_model, device, type)
    vision = process_img(ls.replace('local_shape','depth').replace('png','npy'), vision_transforms, feature_extractor, vision_model, device, type, vision=True)
    all_tactile.append(tactile.reshape(1,-1))
    all_vision.append(vision.reshape(1,-1))

import pdb; pdb.set_trace()
from sklearn.manifold import TSNE
tsne_tactile = TSNE(n_components=3,perplexity=50).fit_transform(torch.cat(all_tactile,dim=0).cpu().detach().numpy())
tsne_vision = TSNE(n_components=3,perplexity=50).fit_transform(torch.cat(all_vision,dim=0).cpu().detach().numpy())

from sklearn.preprocessing import normalize
tsne_tactile = normalize(tsne_tactile,axis=1)
tsne_vision = normalize(tsne_vision,axis=1)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
selected_colors = get_every_other_color('tab20')
custom_cmap = create_custom_colormap(selected_colors)
scatter = ax.scatter(tsne_tactile[:,0],tsne_tactile[:,1], tsne_tactile[:,2], c=classes,cmap=custom_cmap)
plt.legend(handles=scatter.legend_elements()[0], labels=['Pin', 'Aux', 'Stud', 'USB'])

selected_colors = get_every_other_color('tab20',light=True)
custom_cmap = create_custom_colormap(selected_colors)
scatter = ax.scatter(tsne_vision[:,0],tsne_vision[:,1], tsne_vision[:,2], c=classes,cmap=custom_cmap)

#cursor = mplcursors.cursor()
#cursor.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f})"))
plt.show() #savefig('test')

#plt.legend(handles=scatter.legend_elements()[0], labels=['Pin', 'Aux', 'Stud', 'USB'])
#plt.show() #savefig('test')
import pdb; pdb.set_trace()

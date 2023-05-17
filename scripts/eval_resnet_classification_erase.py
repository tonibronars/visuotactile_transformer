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

from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomErasing
from visuotactile_transformer.classes.dataset_classification import VisuotactileDataset, Normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print('device: ', device)

vision_name = 'vision_resnet_classification_erase_big' 
visuotactile_name = 'visuotactile_resnet_classification_erase_big' 

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

# Load checkpoint
'''
if tactile_only: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/tactile_classification/visuotactile_model_0250.pth',map_location='cpu')
elif vision_only: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/vision_classification/visuotactile_model_best.pth',map_location='cpu')
#else: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/classification/visuotactile_model_0190.pth',map_location='cpu')
else: visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/resnet_classification_ang/visuotactile_model_best.pth',map_location='cpu')
'''
visuotactile_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/{}/visuotactile_model_best.pth'.format(visuotactile_name),map_location='cpu')
visuotactile_model.eval()

vision_model = torch.load(os.environ['HOME'] + '/src/visuotactile_transformer/models/{}/visuotactile_model_best.pth'.format(vision_name),map_location='cpu')
vision_model.eval()

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
list_shapes = list_shapes[:50]
true_class = true_class[:50]

#list_shapes = np.random.permutation(list_shapes)

transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),])

vision_transforms =  Compose([
        Grayscale(num_output_channels=3),
        #CenterCrop((random_crop_size, random_crop_size)),
        Resize((224, 224)),
        ToTensor(),
        Normalize(),
        #Grayscale(num_output_channels=3),
        RandomErasing(scale=(0.3, 0.3), value=np.random.rand()*0.05 + 0.6, p=1),
        #AddGaussianNoise(0., 0.01)
        ])


def process_img(path, vision_transforms, transforms, vision_model, visuotactile_model, device):
    # load img and transfrom to ViT size
    with torch.no_grad():
        vision_img = Image.fromarray(np.uint8(np.load(path.replace('local_shape','depth').replace('png','npy'))*255))
        img = Image.open(path)
        vision_img = vision_transforms(vision_img)
        img = transforms(img)
        vision_img = torch.tensor(vision_img).float()
        img = torch.tensor(img).float()
        # pass through model
        vision_feat = vision_img.to(device)
        feat = img.to(device)
        vision_feat = Variable(vision_feat)
        feat = Variable(feat)
        vision_output  = vision_model(torch.unsqueeze(vision_feat,0))
        visuotactile_output  = visuotactile_model(torch.unsqueeze(vision_feat,0), torch.unsqueeze(feat,0))
        return vision_output, visuotactile_output


vision_predicted_class = []
visuotactile_predicted_class = []
for i, ls in enumerate(list_shapes):
    print(ls)
    vision_output, visuotactile_output = process_img(ls, vision_transforms, transforms, vision_model, visuotactile_model, device)
    vision_predicted_class.append(np.argmax(torch.nn.Softmax()(vision_output)).item())
    visuotactile_predicted_class.append(np.argmax(torch.nn.Softmax()(visuotactile_output)).item())
    print('true class ', true_class[i])
    print('vision predicted class ', vision_predicted_class[-1])
    print('visuotactile predicted class ', visuotactile_predicted_class[-1])
print('vision fraction classified correctly: ', len(np.where(np.array(vision_predicted_class) == np.array(true_class))[0])/len(list_shapes))
print('visuotactile fraction classified correctly: ', len(np.where(np.array(visuotactile_predicted_class) == np.array(true_class))[0])/len(list_shapes))
np.save(os.environ['HOME'] + '/src/visuotactile_transformer/eval/predicted_class_{}.npy'.format(vision_name),vision_predicted_class)
np.save(os.environ['HOME'] + '/src/visuotactile_transformer/eval/predicted_class_{}.npy'.format(visuotactile_name),visuotactile_predicted_class)
np.save(os.environ['HOME'] + '/src/visuotactile_transformer/eval/true_class_{}.npy'.format(visuotactile_name),true_class)
np.save(os.environ['HOME'] + '/src/visuotactile_transformer/eval/true_class_{}.npy'.format(vision_name),true_class)
import pdb; pdb.set_trace()

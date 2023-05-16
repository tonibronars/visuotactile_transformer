import numpy as np
import torch
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomRotation, CenterCrop, RandomErasing

class Morphological(object):
    def __init__(self, prob=0.4):
        self.prob = prob
        
    def __call__(self, tensor):
        p = np.random.random()
        if tensor.shape[0] < 4:
          tensor = torch.unsqueeze(tensor, 0)
        size = np.random.randint(1,5)
        kernel = torch.ones(size, size)
        if p < self.prob:
          return tensor
        elif p < (1 + (1-self.prob)/4):
          transformed_img = kornia.morphology.dilation(tensor, kernel)
        elif p < (1 + 2*(1-self.prob)/4):
          transformed_img = kornia.morphology.erosion(tensor, kernel)
        elif p < (1 + 3*(1-self.prob)/4):
          transformed_img = kornia.morphology.opening(tensor, kernel)
        else:
          transformed_img = kornia.morphology.closing(tensor, kernel)
        return transformed_img
    
    def __repr__(self):
        return self.__class__.__name__ + '(prob={0})'.format(self.prob)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Normalize(object):
    def __init__(self):
        pass
        
    def __call__(self, tensor):
        return (tensor - tensor.min())/(tensor.max() - tensor.min())
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class VisuotactileDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.data_path = 'visuotactile_transformer/data/'
        self.data_folders = ['pin_big_subset_ang/', 'aux_big_ang/', 'stud_big_ang/', 'usb_big_ang/']
        #self.data_folders = ['pin_big_subset/', 'aux_big/', 'stud_big/', 'usb_big/']
        self.img_filenames = []
        self.img_classes = []
        for i,df in enumerate(self.data_folders):
            self.img_filenames.extend(sorted(glob.glob(self.data_path + df + 'local_shape_*_1.png')))
            self.img_classes.extend([i] * len(glob.glob(self.data_path + df + 'local_shape*1.png')))
        print('length of visuotactile dataset: ', len(self.img_filenames))
        self.transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),
])

        self.tactile_transforms = Compose([
          RandomRotation(5, fill=255),
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),
          #Morphological()
])

        random_crop_size = np.random.randint(75, 96)
        self.vision_transforms = Compose([
            Grayscale(num_output_channels=3),
            #CenterCrop((random_crop_size, random_crop_size)),
            Resize((224, 224)),
            ToTensor(),
            Normalize(),
            #Grayscale(num_output_channels=3),
            #RandomErasing(scale=(0.03, 0.08), value=np.random.rand()*0.05 + 0.6),
            #AddGaussianNoise(0., 0.01)
        ])


    def __getitem__(self, idx):
        #print('file :', self.img_filenames[idx])
        tactile = Image.open(self.img_filenames[idx])
        tactile = self.transforms(tactile)
        item = dict()
        item['tactile'] = torch.tensor(tactile).float() #.permute(2, 0, 1).float()

        vision_array = np.load(self.img_filenames[idx].replace('local_shape','depth').replace('png','npy'))
        vision = Image.fromarray(np.uint8(vision_array*255))
        vision = self.vision_transforms(vision)
        item['vision'] = torch.tensor(vision).float() #.permute(2, 0, 1).float()

        item['target'] = self.img_classes[idx]
        return item


    def __len__(self):
        return len(self.img_filenames)

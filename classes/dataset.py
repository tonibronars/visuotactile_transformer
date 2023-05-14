import numpy as np
import torch
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale


class VisuotactileDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.data_path = 'visuotactile_transformer/data/'
        self.data_folders = ['pin_big_subset/', 'aux_big/', 'stud_big/', 'usb_big/']
        self.img_filenames = []
        for df in self.data_folders:
            self.img_filenames.extend(sorted(glob.glob(self.data_path + df + 'local_shape_*_1.png')))
        print('length of visuotactile dataset: ', len(self.img_filenames))
        self.transforms = Compose([
          Grayscale(num_output_channels=3),
          Resize((224, 224)),
          ToTensor(),
])

    def __getitem__(self, idx):
        #tactile = cv2.imread(self.img_filenames[idx])
        #tactile = cv2.cvtColor(tactile, cv2.COLOR_BGR2RGB)
        tactile = Image.open(self.img_filenames[idx])
        tactile = self.transforms(tactile)
        item = dict()
        item['tactile'] = torch.tensor(tactile).float() #.permute(2, 0, 1).float()

        #vision = cv2.imread(self.img_filenames[idx].replace('local_shape','depth').replace('png','npy'))
        #vision = cv2.cvtColor(vision, cv2.COLOR_BGR2RGB)
        vision_array = np.load(self.img_filenames[idx].replace('local_shape','depth').replace('png','npy'))
        vision = Image.fromarray(np.uint8(vision_array*255))
        vision = self.transforms(vision)
        item['vision'] = torch.tensor(vision).float() #.permute(2, 0, 1).float()
        
        return item


    def __len__(self):
        return len(self.img_filenames)

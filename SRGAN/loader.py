import glob
import random
import os
import numpy as np
import re

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class ImageDataset(Dataset):
    def __init__(self, root, text_file_path, shape):
        height, width = shape
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width), Image.BICUBIC),
                # transforms.Resize((height*2, width*2), Image.BICUBIC),
                # transforms.CenterCrop((height, width)),

                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform_lr = transforms.Compose(
            [
                transforms.Resize((height//4, width//4), Image.BICUBIC),
                # transforms.Resize((height*2, width*2), Image.BICUBIC),
                # transforms.CenterCrop((height, width)),
                # transforms.Resize((height//4, width//4), Image.BICUBIC),

                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.root=root
        with open(text_file_path, 'r') as f:
            self.files = [x.strip() for x in f.readlines()]


    def __getitem__(self, index):
        file_name=re.split(r'_|\.', self.files[index % len(self.files)])
        path=self.root+'/source'
        gt_path=self.root
        if file_name[2].startswith('OH') or file_name[2].startswith('IH'): #밝은 조도
            path+='/bright_'
            gt_path+='/bright_GT/'
        else:
            path+='/dark_'
            gt_path+='/dark_GT/'
        path+=file_name[-2]+'/'
        
        clean_img = Image.open(gt_path+self.files[index % len(self.files)][:-6]+'GT.jpg')
        # img = Image.open(gt_path+self.files[index % len(self.files)][:-6]+'GT.jpg')
        img = Image.open(path+self.files[index % len(self.files)])
        img_gt = self.transform(clean_img)
        img_lr = self.transform_lr(img)

        return {"lr": img_lr, "gt": img_gt}

    def __len__(self):
        return len(self.files)




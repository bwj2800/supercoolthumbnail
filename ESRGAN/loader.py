
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    def __init__(self, root, text_file_path, shape):
        height, width = shape
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                # transforms.Resize((height, width), Image.BICUBIC),
                transforms.Resize((height*2, width*2), Image.BICUBIC),
                transforms.CenterCrop((height, width)),

                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform_lr = transforms.Compose(
            [
                # transforms.Resize((height//4, width//4), Image.BICUBIC),
                transforms.Resize((height*2, width*2), Image.BICUBIC),
                transforms.CenterCrop((height, width)),
                transforms.Resize((height//4, width//4), Image.BICUBIC),

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
        label_path=self.root+'/label'
        gt_path=self.root
        gt_label_path=self.root+'/label'
        if file_name[2].startswith('OH') or file_name[2].startswith('IH'): #밝은 조도
            path+='/bright_'
            gt_path+='/bright_GT/'
            label_path+='/bright_'
            gt_label_path+='/bright_GT/'
        else:
            path+='/dark_'
            gt_path+='/dark_GT/'
            label_path+='/dark_'
            gt_label_path+='/dark_GT/'

        path+=file_name[-2]+'/'
        label_path+=file_name[-2]+'/'
        
        clean_img = Image.open(gt_path+self.files[index % len(self.files)][:-6]+'GT.jpg')
        # img = Image.open(gt_path+self.files[index % len(self.files)][:-6]+'GT.jpg')
        img = Image.open(path+self.files[index % len(self.files)])
        
        # Clean 이미지와 바운딩 박스의 중점을 일치시키는 작업 추가
        with open(gt_label_path+self.files[index % len(self.files)][:-6]+'GT.json', 'r', encoding='utf-8-sig') as json_file:
            gt_data = json.load(json_file)
        type_value = gt_data["Learning_Data_Info."]["Annotations"][0]["Type_value"]
        gt_x_coords = [type_value[i] for i in range(0, len(type_value), 2)]
        gt_y_coords = [type_value[i] for i in range(1, len(type_value), 2)]
        gt_bbox_center = (sum(gt_x_coords) / 4, sum(gt_y_coords) / 4)

        with open(label_path+self.files[index % len(self.files)][:-3]+'json', 'r', encoding='utf-8-sig') as json_file:
            data = json.load(json_file)
        type_value = data["Learning_Data_Info."]["Annotations"][0]["Type_value"]
        x_coords = [type_value[i] for i in range(0, len(type_value), 2)]
        y_coords = [type_value[i] for i in range(1, len(type_value), 2)]
        bbox_center = (sum(x_coords) / 4, sum(y_coords) / 4)

        translation_vector = (gt_bbox_center[0] - bbox_center[0], gt_bbox_center[1] - bbox_center[1])
        clean_img = clean_img.transform(clean_img.size, Image.AFFINE, (1, 0, translation_vector[0], 0, 1, translation_vector[1]))
        

        img_gt = self.transform(clean_img)
        img_lr = self.transform_lr(img)

        return {"lr": img_lr, "gt": img_gt, "file_name": self.files[index % len(self.files)]}

    def __len__(self):
        return len(self.files)



class TestImageDataset(Dataset):
    def __init__(self, root, text_file_path, shape):
        height, width = shape
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                # transforms.Resize((height, width), Image.BICUBIC),
                transforms.Resize((height*2, width*2), Image.BICUBIC),
                transforms.CenterCrop((height, width)),

                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform_lr = transforms.Compose(
            [
                # transforms.Resize((height//4, width//4), Image.BICUBIC),
                transforms.Resize((height*2, width*2), Image.BICUBIC),
                transforms.CenterCrop((height, width)),
                transforms.Resize((height//4, width//4), Image.BICUBIC),

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
        label_path=self.root+'/label'
        gt_path=self.root
        gt_label_path=self.root+'/label'
        if file_name[2].startswith('OH') or file_name[2].startswith('IH'): #밝은 조도
            path+='/bright_'
            gt_path+='/bright_GT/'
            label_path+='/bright_'
            gt_label_path+='/bright_GT/'
        else:
            path+='/dark_'
            gt_path+='/dark_GT/'
            label_path+='/dark_'
            gt_label_path+='/dark_GT/'

        path+=file_name[-2]+'/'
        label_path+=file_name[-2]+'/'
        
        clean_img = Image.open(gt_path+self.files[index % len(self.files)][:-6]+'GT.jpg')
        # img = Image.open(gt_path+self.files[index % len(self.files)][:-6]+'GT.jpg')
        img = Image.open(path+self.files[index % len(self.files)])

        
        # Clean 이미지와 바운딩 박스의 중점을 일치시키는 작업 추가
        with open(gt_label_path+self.files[index % len(self.files)][:-6]+'GT.json', 'r', encoding='utf-8-sig') as json_file:
            gt_data = json.load(json_file)
        type_value = gt_data["Learning_Data_Info."]["Annotations"][0]["Type_value"]
        gt_x_coords = [type_value[i] for i in range(0, len(type_value), 2)]
        gt_y_coords = [type_value[i] for i in range(1, len(type_value), 2)]
        gt_bbox_center = (sum(gt_x_coords) / 4, sum(gt_y_coords) / 4)

        with open(label_path+self.files[index % len(self.files)][:-3]+'json', 'r', encoding='utf-8-sig') as json_file:
            data = json.load(json_file)
        type_value = data["Learning_Data_Info."]["Annotations"][0]["Type_value"]
        x_coords = [type_value[i] for i in range(0, len(type_value), 2)]
        y_coords = [type_value[i] for i in range(1, len(type_value), 2)]
        bbox_center = (sum(x_coords) / 4, sum(y_coords) / 4)

        translation_vector = (gt_bbox_center[0] - bbox_center[0], gt_bbox_center[1] - bbox_center[1])
        clean_img = clean_img.transform(clean_img.size, Image.AFFINE, (1, 0, translation_vector[0], 0, 1, translation_vector[1]))
        
        img_gt = self.transform(clean_img)
        img_lr = self.transform_lr(img)

        return {"lr": img_lr, "gt": img_gt, "file_name": self.files[index % len(self.files)]}

    def __len__(self):
        return len(self.files)
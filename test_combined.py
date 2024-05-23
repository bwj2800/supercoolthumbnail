import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torchvision import models
from ESRGAN.esrgan import GeneratorRRDB
from loader_combined import denormalize, TestImageDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from torcheval.metrics.functional import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import numpy as np

RESNET_MODEL_NAME = "resnet34_2"
RESNET_MODEL_PATH = './result/resnet/saved_model/'+RESNET_MODEL_NAME+'.h5'
FOCUS_MODEL_NAME='focus_model2'
focus_checkpoint='generator_54'
SHAKEN_MODEL_NAME='shaken_model2'
shaken_checkpoint='generator_44'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default="./datasets/validation", required=False, help="Path to image")
parser.add_argument("--dataset_txt_path", type=str, default="./datasets/new_val_all.txt", required=False, help="Path to image")
parser.add_argument("--output_path", type=str, default="./result/combined/images/test/", required=False, help="Path to image")
parser.add_argument("--focus_checkpoint_model", type=str, default="./result/esrgan/saved_model/"+FOCUS_MODEL_NAME+"/"+focus_checkpoint+".pth", required=False, help="Path to checkpoint model")
parser.add_argument("--shaken_checkpoint_model", type=str, default="./result/esrgan/saved_model/"+SHAKEN_MODEL_NAME+"/"+shaken_checkpoint+".pth", required=False, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels", required=False)
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G", required=False)
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation", required=False)
parser.add_argument("--hr_height", type=int, default=512, help="high res. image height", required=False)
parser.add_argument("--hr_width", type=int, default=1024, help="high res. image width", required=False)
opt = parser.parse_args()

# 클래스 매핑 딕셔너리
class_mapping = {
    'FF.jpg': 0,
    'RF.jpg': 0,
    'RL.jpg': 1,
    'UD.jpg': 1,
}

# Tensor를 Numpy 배열로 변환하는 함수
def tensor_to_numpy(tensor):
    # Tensor가 GPU에 있다면 CPU로 이동
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Tensor에서 Numpy 배열로 변환
    return tensor.detach().numpy()  # `.detach()`를 추가하여 gradient 정보를 제거

os.makedirs(opt.output_path, exist_ok=True)
f=open('./result/combined/result.txt','a')

# GPU 사용 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet model
num_classes=2
resnet_model = models.resnet34(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH))
resnet_model.to(device)
resnet_model.eval()

# Define generator model and load model checkpoint
shaken_generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
shaken_generator.load_state_dict(torch.load(opt.shaken_checkpoint_model))
shaken_generator.eval()

focus_generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
focus_generator.load_state_dict(torch.load(opt.focus_checkpoint_model))
focus_generator.eval()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    TestImageDataset(root=opt.dataset_root, text_file_path = opt.dataset_txt_path, shape=(opt.hr_height, opt.hr_width)),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu,
)


correct = 0
total = 0
psnr_val = []
ssim_val = []
# Upsample image
with torch.no_grad():
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["gt"].type(Tensor))
        imgs_rs = Variable(imgs["rs"].type(Tensor))
        filename = imgs["file_name"][0]

        batches_done = i

        # Classify the type of image with Resnet model
        outputs = resnet_model(imgs_rs)
        _, predicted = torch.max(outputs, 1)

        last_element = filename.split('_')[-1]

        if last_element in class_mapping:
            label = class_mapping[last_element]
        else: # class_mapping에 없는 경우, 예외 처리
            print(f"{last_element}는 class_mapping에 없습니다.")
            label = None  # 또는 적절한 예외 처리
        
        total += 1
        if (predicted == label): correct+=1

        if predicted==0: # classified as out-of-focus image
            gen_hr = focus_generator(imgs_lr)
        else: # classified as shaken image
            gen_hr = shaken_generator(imgs_lr)
            
        # calculate PSNR
        psnr=peak_signal_noise_ratio(denormalize(imgs_hr), denormalize(gen_hr))
        psnr_val.append(psnr)

        # calculate SSIM
        np_imgs_hr = tensor_to_numpy(imgs_hr)
        np_gen_hr = tensor_to_numpy(gen_hr)
        
        np_imgs_hr = np.squeeze(np_imgs_hr, axis=0)
        np_gen_hr = np.squeeze(np_gen_hr, axis=0)

        np_imgs_hr = np.transpose(np_imgs_hr, (1, 2, 0))
        np_gen_hr = np.transpose(np_gen_hr, (1, 2, 0))

        ssim_value = ssim(np_imgs_hr, np_gen_hr, channel_axis=2, data_range=1.0)

        ssim_val.append(ssim_value)

        print(
            "[Batch %d/%d] [PSNR: %f] [SSIM: %f] %s"
            % (
                i,
                len(dataloader),
                psnr,
                ssim_value,
                imgs["file_name"][0]
            )
        )
        # Save image
        save_image((gen_hr).cpu(), opt.output_path+f"/sr-"+filename)


test_accuracy = (correct / total) * 100
print("Accuracy:", test_accuracy)
print("Average PSNR:",(sum(psnr_val) / len(psnr_val)).item())
print("Average SSIM:", (sum(ssim_val) / len(ssim_val)).item())

RESULT = "\nAccuracy: {:.5f} Average PSNR: {:.5f} Average SSIM: {:.5f}".format(
    test_accuracy, sum(psnr_val) / len(psnr_val), sum(ssim_val) / len(ssim_val))
f.write(RESULT)
f.close()
from esrgan import GeneratorRRDB
from loader import denormalize, TestImageDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from torcheval.metrics.functional import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm

MODEL_NAME='all_model2'
checkpoint=461
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default="../datasets/validation", required=False, help="Path to image")
parser.add_argument("--dataset_txt_path", type=str, default="../datasets/new_val_all.txt", required=False, help="Path to image")
parser.add_argument("--output_path", type=str, default="../result/esrgan/images/test/"+MODEL_NAME, required=False, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default="../result/esrgan/saved_model/"+MODEL_NAME+"/generator_"+str(checkpoint)+".pth", required=False, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels", required=False)
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G", required=False)
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation", required=False)
parser.add_argument("--hr_height", type=int, default=512, help="high res. image height", required=False)
parser.add_argument("--hr_width", type=int, default=1024, help="high res. image width", required=False)
opt = parser.parse_args()

# Tensor를 Numpy 배열로 변환하는 함수
def tensor_to_numpy(tensor):
    # Tensor가 GPU에 있다면 CPU로 이동
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Tensor에서 Numpy 배열로 변환
    return tensor.detach().numpy()  # `.detach()`를 추가하여 gradient 정보를 제거

os.makedirs(opt.output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT=""
best_psnr=0
best_psnr_checkpoint=0
best_ssim=0
best_ssim_checkpoint=0

for c in tqdm(range(1)):
    # Define model and load model checkpoint
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    # generator.load_state_dict(torch.load("../result/esrgan/saved_model/"+MODEL_NAME+"/generator_"+str(c)+".pth"))
    generator.load_state_dict(torch.load("../result/esrgan/saved_model/"+MODEL_NAME+"/generator_49.pth"))
    generator.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    dataloader = DataLoader(
        TestImageDataset(root=opt.dataset_root, text_file_path = opt.dataset_txt_path, shape=(opt.hr_height, opt.hr_width)),
        batch_size=1,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    psnr_val = []
    ssim_val = []

    # Upsample image
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):

            batches_done = i

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["gt"].type(Tensor))

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

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

            # print(
            #     "[Batch %d/%d] [PSNR: %f] [SSIM: %f] %s"
            #     % (
            #         i,
            #         len(dataloader),
            #         psnr,
            #         ssim_value,
            #         imgs["file_name"][0]
            #     )
            # )

            # Save image
            save_image((gen_hr).cpu(), opt.output_path+f"/sr-"+imgs["file_name"][0])

    avg_psnr=sum(psnr_val) / len(psnr_val)
    avg_ssim=sum(ssim_val) / len(ssim_val)
    print("Average PSNR:", avg_psnr)
    print("Average SSIM:", avg_ssim)

    RESULT += "\n{} Average PSNR: {:.5f} Average SSIM: {:.5f}".format(c, avg_psnr, avg_ssim)

    if avg_psnr>best_psnr:
        best_psnr=avg_psnr
        best_psnr_checkpoint=c
    if avg_ssim>best_ssim:
        best_ssim=avg_ssim
        best_ssim_checkpoint=c

RESULT += "\nBest Average PSNR at generator_{}: {:.5f}".format(best_psnr_checkpoint, best_psnr)
RESULT += "\nBest Average SSIM at generator_{}: {:.5f}".format(best_ssim_checkpoint, best_ssim)

f=open('../result/esrgan/'+MODEL_NAME+'.txt','a')
f.write(RESULT)
f.close()
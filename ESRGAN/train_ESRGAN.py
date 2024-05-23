"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np

from torchvision.utils import save_image, make_grid


from torch.utils.data import DataLoader
from torch.autograd import Variable

from esrgan import *
from loader import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from torcheval.metrics.functional import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import numpy as np
torch.cuda.empty_cache()

SAVE_IMAGE_DIR="../result/esrgan/images/training/focus_model2"
SAVE_MODEL_DIR="../result/esrgan/saved_model/focus_model2"
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")

parser.add_argument("--epoch", type=int, default=492, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--dataset_txt_path", type=str, default="../datasets/new_focus.txt", help="path of the txt file of training dataset")
parser.add_argument("--val_dataset_txt_path", type=str, default="../datasets/new_val_focus.txt", help="path of the txt file of training dataset")
parser.add_argument("--dataset_root", type=str, default="../datasets/training", help="root path of train dataset")
parser.add_argument("--val_dataset_root", type=str, default="../datasets/validation", help="root path of train dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--g_lr", type=float, default=0.0002, help="adam: learning rate of generator")
parser.add_argument("--d_lr", type=float, default=0.0002, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
# print(opt)

# Tensor를 Numpy 배열로 변환하는 함수
def tensor_to_numpy(tensor):
    # Tensor가 GPU에 있다면 CPU로 이동
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Tensor에서 Numpy 배열로 변환
    return tensor.detach().numpy()  # `.detach()`를 추가하여 gradient 정보를 제거

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("../result/esrgan/saved_model/focus_model2/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("../result/esrgan/saved_model/focus_model2/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset(root=opt.dataset_root, text_file_path = opt.dataset_txt_path, shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    TestImageDataset(root=opt.val_dataset_root, text_file_path = opt.val_dataset_txt_path, shape=(opt.hr_height, opt.hr_width)),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

best_psnr=0

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["gt"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            )
        )

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr, imgs_hr), -1))
            # save_image(img_grid, "images/esrgan/%d.png" % batches_done, nrow=1, normalize=False)
            save_image(img_grid, SAVE_IMAGE_DIR+"/epoch%d_%d.png" % (epoch, batches_done), nrow=1, normalize=False)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), SAVE_MODEL_DIR+"/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), SAVE_MODEL_DIR+"/discriminator_%d.pth" %epoch)
    

    #### Evaluation ####
    psnr_val = []
    ssim_val = []

    with torch.no_grad():
        for i, imgs in enumerate(val_dataloader):
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


        avg_psnr=sum(psnr_val) / len(psnr_val)
        avg_ssim=sum(ssim_val) / len(ssim_val)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(generator.state_dict(), SAVE_MODEL_DIR+"/best_generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), SAVE_MODEL_DIR+"/best_discriminator_%d.pth" %epoch)

            print("saved best model at [epoch %d PSNR: %.4f SSIM: %.4f]" % (epoch, avg_psnr, avg_ssim))
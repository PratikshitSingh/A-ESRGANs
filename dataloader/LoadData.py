import os
import torch
import numpy as np
import random
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import paired_random_crop, augment
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from PIL import ImageFile, Image
from torch.utils import data
from torch.nn import functional as F
import cv2
import math


data_path = 'imgs'
def get_all_file(target_dir,subfix=None): # just for this project
  g = os.walk(target_dir)
  data = []
  for path,dir_list,file_list in sorted(g):
    data += sorted(file_list)
  data = [os.path.join(target_dir,i) for i in data]
  if subfix != None:
    data = [i for i in data if i.endswith(subfix)]
  return data

# @title Dataset
class LoadData(data.Dataset):

    def __init__(self, split="train", preload=True, data_dir=data_path, num_data=None, image=None, crop=True):
        assert (split in ["train", "val", "test"])
        # if split == "train":
        #   self.img_dir = os.path.join(img_dir,type_2)
        #   self.data_dir = os.path.join(data_dir,data_type)
        # else:
        #   self.img_dir = os.path.join(img_dir,split,type_2)
        #   self.data_dir = os.path.join(data_dir,split,data_type)
        # print(self.img_dir)
        # print(self.data_dir)

        # self.split = split
        # self.data = get_all_file(self.img_dir,subfix="png", img= type_2 == "")
        # self.mask_data = get_all_file(self.data_dir,subfix="png", img=False)
        self.crop = crop
        self.img_dir = os.path.join(data_dir, split)
        self.images = get_all_file(self.img_dir, subfix="png")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # insert dataloading here
        # self.img = image #temp testing var

        self.preload = preload

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

        if preload:
            self.images = list(map(Image.open, self.images[:num_data]))

    def __len__(self):
        return len(self.images)

    def init_kernels(self):
        blur_kernel_size = 21
        kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        sinc_prob = 0.1
        blur_sigma = [0.2, 3]
        blur_sigma2 = [0.2, 1.5]
        betag_range = [0.5, 4]
        betap_range = [1, 2]
        final_sinc_prob = 0.8
        # --------------kernel1-------------------
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma, [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        # --------------kernel2-------------------
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma2,
                blur_sigma2, [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        # --------------sinc-------------------
        if np.random.uniform() < final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            pulse_tensor = torch.zeros(21, 21).float()
            pulse_tensor[10, 10] = 1
            sinc_kernel = pulse_tensor

        kernel = torch.FloatTensor(kernel).to(self.device)
        kernel2 = torch.FloatTensor(kernel2).to(self.device)
        sinc_kernel = torch.FloatTensor(sinc_kernel).to(self.device)
        return kernel, kernel2, sinc_kernel

    def transform(self, img, kernel1, kernel2, sinc, crop=True):
        # print(img.size())
        ori_h = img.shape[1]
        ori_w = img.shape[2]
        img = img.unsqueeze(0).float()
        usm_sharpener = USMSharp().to(self.device)
        jpeger = DiffJPEG(differentiable=False).to(self.device)
        img_usm = usm_sharpener(img)

        gauss_noise_prob = 0.5
        resize_range1 = 0.5
        resize_range2 = 0.25
        sigma_range = (1, 30)
        pscale = (0.05, 3)
        scale_opt = 4

        out = filter2D(img_usm.clone(), kernel1)
        updown_type = random.choices(['up', 'down', 'keep'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, 1 + resize_range1)
        elif updown_type == 'down':
            scale = np.random.uniform(1 - resize_range1, 1)
        else:
            scale = 1

        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        gray_noise_prob = 0.33
        if np.random.uniform() < gauss_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=sigma_range,
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=pscale,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(30, 95)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)

        blur2_prob = 0.333
        if np.random.uniform() < blur2_prob:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, 1 + resize_range2)
        elif updown_type == 'down':
            scale = np.random.uniform(1 - resize_range2, 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])

        out = F.interpolate(
            out, size=(int(ori_h / scale_opt * scale), int(ori_w / scale_opt * scale)), mode=mode)
        # noise
        if np.random.uniform() < gauss_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=sigma_range,
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=pscale,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_opt, ori_w // scale_opt), mode=mode)
            out = filter2D(out, sinc)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(30, 95)
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(30, 95)
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_opt, ori_w // scale_opt), mode=mode)
            out = filter2D(out, sinc)

        gt_size = 256
        gt = img_usm
        lq = out
        if crop:
            gt, lq = paired_random_crop(img_usm, out, gt_size, scale_opt)
        return gt, lq

    def __getitem__(self, idx):
        if self.preload:
            img = self.images[idx]
        else:
            # img = list(map(cv2.imread, self.images[idx]))
            img_gt = cv2.imread(self.images[idx])
        img_gt = augment(img_gt, True, False)
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if h > crop_pad_size or w > crop_pad_size:
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        img_gt = np.transpose(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), (2, 0, 1))
        img = torch.from_numpy(img_gt) / 255.
        # print(img.max(), img.min())
        img = img.to(self.device)
        kernel1, kernel2, sinc = self.init_kernels()
        gt, lq = self.transform(img, kernel1, kernel2, sinc)
        return gt.squeeze(), lq.squeeze()

if __name__ == '__main__':
    LD = LoadData()
import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from . import register
from ..utils import to_pixel_samples, to_pixel_samples_nb
import torch.nn.functional as F
import torch.nn as nn

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

import torchvision.transforms as T

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

def resize_event(event, size):
    return F.interpolate(event.unsqueeze(0), size = size, mode='nearest').squeeze(0)

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # s = random.uniform(self.scale_min, self.scale_max)

        # w_hr = 180
        # w_lr = 45
        w_hr = 512
        w_lr = 128
        w_llr = 32
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(resize_fn(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr], w_lr))
            crop_lr.append(resize_fn(crop_hr[i], w_llr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_llr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_llr)

        crop_hr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_hr_start_event = augment(crop_hr_start_event)
            crop_hr_end_event = augment(crop_hr_end_event)

            # crop_hr_event = augment(crop_hr_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('sr-super-downsampled')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # s = random.uniform(self.scale_min, self.scale_max)

        # w_hr = 180
        # w_lr = 45

        w_hr = 512
        w_lr = 128

        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)

        crop_hr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_hr_start_event = augment(crop_hr_start_event)
            crop_hr_end_event = augment(crop_hr_end_event)

            # crop_hr_event = augment(crop_hr_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('vfi-train')
class VFI_TRAIN(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 512
        w_lr = 128
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            # crop_hr_event = augment(crop_hr_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : crop_lr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('vfi-train')
class VFI_TRAIN(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 512
        w_lr = 128
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            # crop_hr_event = augment(crop_hr_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : crop_lr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('sr-val-downsampled')
class SRValDownsampled(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # s = random.uniform(self.scale_min, self.scale_max)

        w_hr = 384
        w_lr = 96
        # x0 = random.randint(0, img[0].shape[-2] - w_hr)
        # y0 = random.randint(0, img[0].shape[-1] - w_hr)
        x0 = (img[0].shape[-2] - w_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('sr-test-downsampled')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # s = random.uniform(self.scale_min, self.scale_max)

        w_hr = 1280
        h_hr = 640 #720
        w_lr = 320
        h_lr = 160 #180
        # x0 = random.randint(0, img[0].shape[-2] - w_hr)
        # y0 = random.randint(0, img[0].shape[-1] - w_hr)
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('vfi-test')
class VFI_TEST(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 1280
        h_hr = 720
        w_lr = 320
        h_lr = 180
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : crop_lr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
        }

@register('sr-liif-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            # crop_hr_event = augment(crop_hr_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            # 'gt' : crop_hr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
            # 'inp_event' : crop_lr_event
        }

@register('sr-liif-test-downsampled')
class SRliiftestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 1280
        h_hr = 720
        w_lr = 320
        h_lr = 180
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            # 'gt' : crop_hr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
            # 'inp_event' : crop_lr_event
        }

@register('sr-implicit-downsampled-timelens')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # s = random.uniform(self.scale_min, self.scale_max)

        w_hr = 512
        w_lr = 128
        w_llr = 32
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(resize_fn(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr], w_lr))
            crop_lr.append(resize_fn(crop_hr[i], w_llr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_llr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_llr)

        crop_hr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_hr_start_event = augment(crop_hr_start_event)
            crop_hr_end_event = augment(crop_hr_end_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            'time' : time
        }

@register('frame-train-downsampled-timelens')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # img = [data['start'], data['gt'], data['end']]
        img = data['imgs']

        # w_hr = 512
        # w_lr = 128
        w_hr = 256

        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []

        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : crop_hr[1],
            'start' : crop_hr[0],
            'end' : crop_hr[2],
            'start_event' : [],
            'end_event' : [],
            'time' : time
            }

@register('frame-test-downsampled-timelens')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : img[1],
            'start' : img[0],
            'end' : img[2],
            'start_event' : [],
            'end_event' : [],
            'time' : time,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('sr-implicit-downsampled-timelens-super')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # s = random.uniform(self.scale_min, self.scale_max)

        # w_hr = 180
        # w_lr = 45

        w_hr = 512
        w_lr = 128

        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)

        crop_hr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = resize_event(crop_hr_end_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_hr_start_event = augment(crop_hr_start_event)
            crop_hr_end_event = augment(crop_hr_end_event)

            # crop_hr_event = augment(crop_hr_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            'time' : time
            # 'gt': hr_rgb,
            # 'inp_event' : crop_lr_event
            }

@register('sr-test-downsampled-timelens')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 896
        h_hr = 512
        w_lr = 224
        h_lr = 128
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event
        }

@register('dsr-train-downsampled-timelens')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # img = [data['start'], data['gt'], data['end']]
        img = data['imgs']
        start_event = data['start_event']
        rgb = data['rgbs']
        mask = data['masks']

        # w_hr = 512
        # w_lr = 128
        # w_hr = 256
        w_hr = 480

        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        crop_mask = []
        crop_rgb = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_mask.append(mask[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_rgb.append(rgb[i][:, x0: x0 + w_hr, y0: y0 + w_hr])

        crop_lr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_mask[i] = augment(crop_mask[i])
                crop_rgb[i] = augment(crop_rgb[i])

            crop_lr_start_event = augment(crop_lr_start_event)

        time = 1.0
        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : crop_hr[1],
            'start' : crop_hr[0],
            'start_event' : crop_lr_start_event,
            'time' : time,
            'gt_mask': crop_mask[1],
            'gt_rgb': crop_rgb[1],
            'start_mask': crop_mask[0],
            'start_rgb': crop_rgb[0]
            }

@register('dsr-test-downsampled-timelens')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']
        start_event = data['start_event']
        rgb = data['rgbs']
        mask = data['masks']

        time = 1

        # time = (data['frameRange'][1] - data['frameRange'][0])
        # time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : img[1],
            'start' : img[0],
            'start_event' : start_event,
            'time' : time,
            'gt_path': data['gt_path'],
            'depth_scale': data['depth_scale'],
            'gt_mask': mask[1],
            'gt_rgb': rgb[1],
            'start_mask': mask[0],
            'start_rgb': rgb[0]
            # 'hr_total_event' : crop_hr_total_event
        }

@register('blender-e2vid-test')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datas = self.dataset[idx]
        results = []
        for data in datas:
            img = data['imgs']
            start_event = data['start_event']
            rgb = data['rgbs']
            mask = data['masks']
            recon_event = data['recon_event']

            time = 1

            result = {
                'gt' : img[1],
                'start' : img[0],
                'start_event' : start_event,
                'recon_event' : recon_event,
                'time' : time,
                'gt_path': data['gt_path'],
                'depth_scale': data['depth_scale'],
                'gt_mask': mask[1],
                'gt_rgb': rgb[1],
                'start_mask': mask[0],
                'start_rgb': rgb[0]
            }
            results.append(result)
        return results

@register('dsr-train-downsampled-timelens-evimo')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # img = [data['start'], data['gt'], data['end']]
        img = data['imgs']
        start_event = data['start_event']

        # w_hr = 256
        w_hr = 480

        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])

        crop_lr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])

            crop_lr_start_event = augment(crop_lr_start_event)

        time = 1.0
        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : crop_hr[1],
            'start' : crop_hr[0],
            'start_event' : crop_lr_start_event,
            'time' : time,
            }

@register('blender-e2vid')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datas = self.dataset[idx]

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        w_hr = 480

        x0 = random.randint(0, datas[0]['imgs'][0].shape[-2] - w_hr)
        y0 = random.randint(0, datas[0]['imgs'][0].shape[-1] - w_hr)

        results = []
        for data in datas:
            img = data['imgs']
            start_event = data['start_event']
            recon_event = data['recon_event']

            # w_hr = 256

            crop_hr = []
            for i in range(len(img)):
                crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])

            crop_lr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr_recon_event = recon_event[:, x0: x0 + w_hr, y0: y0 + w_hr]

            if self.augment:

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    return x

                # crop_lr = augment(crop_lr)
                for i in range(len(crop_hr)):
                    crop_hr[i] = augment(crop_hr[i])

                crop_lr_start_event = augment(crop_lr_start_event)
                crop_lr_recon_event = augment(crop_lr_recon_event)

            time = 1.0
            result = {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : crop_hr[1],
            'start' : crop_hr[0],
            'start_event' : crop_lr_start_event,
            'recon_event' : crop_lr_recon_event,
            'time' : time,
            'gt_path': data['gt_path'],
            }
            results.append(result)
        return results

@register('blender-e2vid-resize')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datas = self.dataset[idx]

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        # w_hr = 480
        w_hr = random.randint(128, 384)
        H = datas[0]['imgs'][0].shape[-2]      # >>> NEW
        W = datas[0]['imgs'][0].shape[-1]      # >>> NEW

        x0 = random.randint(0, H - w_hr)
        y0 = random.randint(0, W - w_hr)

        results = []
        for data in datas:
            img = data['imgs']
            start_event = data['start_event']
            recon_event = data['recon_event']

            # w_hr = 256

            crop_hr = []
            for i in range(len(img)):
                crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])

            crop_lr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr_recon_event = recon_event[:, x0: x0 + w_hr, y0: y0 + w_hr]

            if self.augment:

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    return x

                # crop_lr = augment(crop_lr)
                for i in range(len(crop_hr)):
                    crop_hr[i] = augment(crop_hr[i])

                crop_lr_start_event = augment(crop_lr_start_event)
                crop_lr_recon_event = augment(crop_lr_recon_event)

            def resize_256(x):
                # x: [C,H,W] -> [1,C,H,W] -> interpolate -> [C,256,256]
                return F.interpolate(x.unsqueeze(0), size=(256, 256),
                                     mode='bilinear', align_corners=False).squeeze(0)

            for i in range(len(crop_hr)):
                crop_hr[i] = resize_256(crop_hr[i])  # >>> NEW
            crop_lr_start_event = resize_256(crop_lr_start_event)  # >>> NEW
            crop_lr_recon_event = resize_256(crop_lr_recon_event)  # >>> NEW

            time = 1.0
            result = {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : crop_hr[1],
            'start' : crop_hr[0],
            'start_event' : crop_lr_start_event,
            'recon_event' : crop_lr_recon_event,
            'time' : time,
            'gt_path': data['gt_path'],
            }
            results.append(result)
        return results

@register('ours-demo-e2vid')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datas = self.dataset[idx]

        results = []
        for data in datas:
            img = data['imgs']
            start_event = data['start_event']
            recon_event = data['recon_event']

            result = {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            # 'gt' : img[1],
            'start' : img[0],
            'start_event' : start_event,
            'recon_event' : recon_event,
            'gt_path': data['gt_path'],
            'data_index': data['index']
            }
            results.append(result)
        return results

@register('dsr-test-downsampled-timelens-ho3d')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']
        start_event = data['start_event']

        x0 = 0
        y0 = 0
        w_hr = 160

        time = 1

        # time = (data['frameRange'][1] - data['frameRange'][0])
        # time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : img[1],
            'start' : img[0],
            'start_event' : start_event,
            'time' : time,
            'gt_path': data['gt_path'],
            }

@register('dsr-test-downsampled-timelens-demo')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']
        start_event = data['start_event']
        end_event = data['end_event']

        path = data['path']

        time = data['time']

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'start' : img[0],
            'end' : img[1],
            'start_event' : start_event,
            'end_event' : end_event,
            'time' : time,
            'path' : path
            # 'hr_total_event' : crop_hr_total_event
        }

@register('dsr-test-downsampled-timelens-time')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']
        hr_img = data['hr_imgs']
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']
        # lr_total_event = data['lr_total_event']

        w_hr = 448
        h_hr = 256
        w_lr = 112
        h_lr = 64

        x0 = (hr_img[0].shape[-2]//4 - h_lr)//2
        y0 = (hr_img[0].shape[-1]//4 - w_lr)//2
        h_x0 = 4 * x0
        h_y0 = 4 * y0

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_lr.append(img[i][:, x0: x0 + h_lr, y0: y0 + w_lr])
        for i in range(len(hr_img)):
            crop_hr.append(hr_img[i][:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr])
        crop_lr_start_event = start_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        crop_lr_end_event = end_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        crop_lr_total_event = total_event[:, x0: x0 + h_lr, y0: y0 + w_lr]

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : torch.stack([crop_hr[0], crop_hr[1], crop_hr[2]]),
            'start' : crop_lr[data['frameRange'][0]],
            'end' : crop_lr[data['frameRange'][2]],
            'lr_gts' : torch.stack(crop_lr[1:]),
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('dsr-train-downsampled-hsergb')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        lr_start_event = data['lr_start_event']
        end_event = data['end_event']
        lr_end_event = data['lr_end_event']
        # total_event = data['total_event']
        # lr_total_event = data['lr_total_event']

        w_hr = 512
        w_lr = 128
        # w_hr = 256
        # w_lr = 64

        x0 = random.randint(0, img[0].shape[-2]//4 - w_lr)
        y0 = random.randint(0, img[0].shape[-1]//4 - w_lr)
        h_x0 = 4 * x0
        h_y0 = 4 * y0

        crop_hr = []
        crop_lr = []

        for i in range(len(img)):
            crop_hr.append(img[i][:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr]
        crop_lr_start_event = lr_start_event[:, x0: x0 + w_lr, y0: y0 + w_lr]
        crop_hr_end_event = end_event[:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr]
        crop_lr_end_event = lr_end_event[:, x0: x0 + w_lr, y0: y0 + w_lr]
        # crop_hr_total_event = total_event[:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr]
        # crop_lr_total_event = lr_total_event[:, x0: x0 + w_lr, y0: y0 + w_lr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            # crop_lr_total_event = augment(crop_lr_total_event)
            crop_hr_start_event = augment(crop_hr_start_event)
            crop_hr_end_event = augment(crop_hr_end_event)
            # crop_hr_total_event = augment(crop_hr_total_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        # print(data['frameRange'], time)

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            # 'total_event' : crop_lr_total_event,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            # 'hr_total_event' : crop_hr_total_event,
            'time' : time
            }

@register('dsr-test-downsampled-hsergb')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # total_event = data['total_event']
        lr_start_event = data['lr_start_event']
        lr_end_event = data['lr_end_event']
        # lr_total_event = data['lr_total_event']

        w_hr = 448
        h_hr = 256
        w_lr = 112
        h_lr = 64

        x0 = (img[0].shape[-2]//4 - h_lr)//2
        y0 = (img[0].shape[-1]//4 - w_lr)//2
        h_x0 = 4 * x0
        h_y0 = 4 * y0

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_start_event = lr_start_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        crop_hr_end_event = end_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_end_event = lr_end_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        # crop_hr_total_event = total_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        # crop_lr_total_event = lr_total_event[:, x0: x0 + h_lr, y0: y0 + w_lr]

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            # 'total_event' : crop_lr_total_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('dsr-test-downsampled-gef')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # total_event = data['total_event']

        w_hr = 760
        h_hr = 720
        w_lr = 190
        h_lr = 180

        x0 = (start_event.shape[-2] - h_lr)//2
        y0 = (start_event.shape[-1] - w_lr)//2
        # h_x0 = 4 * x0
        # h_y0 = 4 * y0

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(resize_fn(img[i], (h_hr, w_hr)))
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_lr_start_event = start_event
        crop_lr_end_event = end_event

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            # 'total_event' : crop_lr_total_event,
            'time' : time,
            # 'hr_start_event' : crop_hr_start_event,
            # 'hr_end_event' : crop_hr_end_event,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('dsr-train-downsampled-ced')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        lr_start_event = data['lr_start_event']
        end_event = data['end_event']
        lr_end_event = data['lr_end_event']
        # total_event = data['total_event']
        # lr_total_event = data['lr_total_event']

        w_hr = 256
        w_lr = 64

        x0 = random.randint(0, img[0].shape[-2]//4 - w_lr)
        y0 = random.randint(0, img[0].shape[-1]//4 - w_lr)
        h_x0 = 4 * x0
        h_y0 = 4 * y0

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr]
        crop_lr_start_event = lr_start_event[:, x0: x0 + w_lr, y0: y0 + w_lr]
        crop_hr_end_event = end_event[:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr]
        crop_lr_end_event = lr_end_event[:, x0: x0 + w_lr, y0: y0 + w_lr]
        # crop_hr_total_event = total_event[:, h_x0: h_x0 + w_hr, h_y0: h_y0 + w_hr]
        # crop_lr_total_event = lr_total_event[:, x0: x0 + w_lr, y0: y0 + w_lr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            # crop_lr_total_event = augment(crop_lr_total_event)
            crop_hr_start_event = augment(crop_hr_start_event)
            crop_hr_end_event = augment(crop_hr_end_event)
            # crop_hr_total_event = augment(crop_hr_total_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        # print(data['frameRange'], time)

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            # 'total_event' : crop_lr_total_event,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            # 'hr_total_event' : crop_hr_total_event,
            'time' : time
            }

@register('dsr-test-downsampled-ced')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        # total_event = data['total_event']
        lr_start_event = data['lr_start_event']
        lr_end_event = data['lr_end_event']
        lr_total_event = data['lr_total_event']

        w_hr = 256
        h_hr = 256
        w_lr = 64
        h_lr = 64

        x0 = (img[0].shape[-2]//4 - h_lr)//2
        y0 = (img[0].shape[-1]//4 - w_lr)//2
        h_x0 = 4 * x0
        h_y0 = 4 * y0

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_start_event = lr_start_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        crop_hr_end_event = end_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_end_event = lr_end_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        # crop_hr_total_event = total_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_total_event = lr_total_event[:, x0: x0 + h_lr, y0: y0 + w_lr]

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'lr_total_event' : crop_lr_total_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('dsr-test-downsampled-ced-viz')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        lr_start_event = start_event
        lr_end_event = end_event

        w_hr = 256
        h_hr = 256
        w_lr = 256
        h_lr = 256

        x0 = (img[0].shape[-2] - h_lr)//2
        y0 = (img[0].shape[-1] - w_lr)//2
        h_x0 = x0
        h_y0 = y0

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr])

        crop_lr = crop_hr
        crop_hr_start_event = start_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_start_event = lr_start_event[:, x0: x0 + h_lr, y0: y0 + w_lr]
        crop_hr_end_event = end_event[:, h_x0: h_x0 + h_hr, h_y0: h_y0 + w_hr]
        crop_lr_end_event = lr_end_event[:, x0: x0 + h_lr, y0: y0 + w_lr]

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('sr-test-downsampled-hqf')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __pad__(self, img):
        m = nn.ZeroPad2d((0, 16, 0, 96))
        # m = nn.ZeroPad2d((0, 16, 0, 32))
        return m(img)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 240
        h_hr = 160
        w_lr = 64
        h_lr = 64

        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(self.__pad__(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr]))
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))

        crop_hr_start_event = self.__pad__(start_event[:, x0: x0 + h_hr, y0: y0 + w_hr])
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = self.__pad__(end_event[:, x0: x0 + h_hr, y0: y0 + w_hr])
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event
        }

@register('sr-test-downsampled-hsergb-viz')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        total_event = data['total_event']

        w_hr = 1280
        h_hr = 1024
        w_lr = 320
        h_lr = 256
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        crop_hr_total_event = total_event[:, x0//4: x0//4 + h_lr, y0//4: y0//4 + w_lr]

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            'total_event' : crop_hr_total_event
        }

@register('sr-test-downsampled-timelens-viz')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 640
        h_hr = 512
        w_lr = 640
        h_lr = 512
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
        crop_lr = crop_hr
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = crop_hr_start_event
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = crop_hr_end_event

        time = (data['frameRange'][1] - data['frameRange'][0])

        # total_event = data['total_event']
        # total_event = resize_event(total_event, (h_lr, w_lr))

        return {
            'start' : crop_lr[0],
            'end' : crop_lr[1],
            'gt' : torch.stack([crop_hr[1], crop_hr[1], crop_hr[1]]),
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event,
            # 'total_event' : total_event
        }

@register('sr-test-downsampled-hsergb')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']

        w_hr = 640
        h_hr = 704
        w_lr = 160
        h_lr = 176
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2
        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            'gt' : torch.stack(crop_hr),
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'lr_gt' : crop_lr[1],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time,
            'hr_start_event' : crop_hr_start_event,
            'hr_end_event' : crop_hr_end_event
        }

@register('sr-liif-temp-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)
        crop_hr_total_event = total_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_total_event = resize_event(crop_hr_total_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_lr_total_event = augment(crop_lr_total_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            # 'gt' : crop_hr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
            # 'inp_event' : crop_lr_event
        }

@register('sr-liif-test-temp-downsampled')
class SRliiftestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']

        w_hr = 1280
        h_hr = 720
        w_lr = 320
        h_lr = 180
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))
        crop_hr_total_event = total_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_total_event = resize_event(crop_hr_total_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-liif-temp-kd-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)
        crop_hr_total_event = total_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_total_event = resize_event(crop_hr_total_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_lr_total_event = augment(crop_lr_total_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            # 'gt' : crop_hr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'lr_gt': crop_lr[1]
        }

@register('sr-liif-test-temp-kd-downsampled')
class SRliiftestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']

        w_hr = 1280
        h_hr = 720
        w_lr = 320
        h_lr = 180
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))
        crop_hr_total_event = total_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_total_event = resize_event(crop_hr_total_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'lr_gt': crop_lr[1]
        }

@register('dsr-train-downsampled-waymo')
class SRSuperDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # img = [data['start'], data['gt'], data['end']]
        img = data['imgs']
        start_event = data['start_event']
        end_event = data['end_event']

        # w_hr = 512
        # w_lr = 128
        w_hr = 256

        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []

        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
        crop_lr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : crop_hr[1],
            'start' : crop_hr[0],
            'end' : crop_hr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'time' : time
            }

@register('dsr-test-downsampled-waymo')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']
        start_event = data['start_event']
        end_event = data['end_event']

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'gt' : img[1],
            'start' : img[0],
            'end' : img[2],
            'start_event' : start_event,
            'end_event' : end_event,
            'time' : time,
            # 'hr_total_event' : crop_hr_total_event
        }

@register('dsr-test-downsampled-waymo-demo')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']
        start_event = data['start_event']
        end_event = data['end_event']

        path = data['path']

        time = data['time']

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'start' : img[0],
            'end' : img[1],
            'start_event' : start_event,
            'end_event' : end_event,
            'time' : time,
            'path' : path
            # 'hr_total_event' : crop_hr_total_event
        }

@register('frame-test-downsampled-timelens-demo')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']

        path = data['path']

        time = data['time']

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'start' : img[0],
            'end' : img[1],
            'time' : time,
            'start_event' : [],
            'end_event' : [],
            'path' : path
            # 'hr_total_event' : crop_hr_total_event
        }

@register('frame-test-downsampled-waymo-demo')
class SRTestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data['imgs']

        path = data['path']

        time = data['time']

        return {
            # 'gt' : torch.stack([crop_hr[data['frameRange'][0]], crop_hr[data['frameRange'][1]], crop_hr[data['frameRange'][2]]]),
            'start' : img[0],
            'end' : img[1],
            'time' : time,
            'start_event' : [],
            'end_event' : [],
            'path' : path
            # 'hr_total_event' : crop_hr_total_event
        }

@register('sr-liif-temp-kd-nb-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img[0].shape[-2] - w_hr)
        y0 = random.randint(0, img[0].shape[-1] - w_hr)

        crop_hr = []
        # crop_hr_event = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + w_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], w_lr))
        crop_hr_start_event = start_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, w_lr)
        crop_hr_end_event = end_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, w_lr)
        crop_hr_total_event = total_event[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr_total_event = resize_event(crop_hr_total_event, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            for i in range(len(crop_hr)):
                crop_hr[i] = augment(crop_hr[i])
                crop_lr[i] = augment(crop_lr[i])
            crop_lr_start_event = augment(crop_lr_start_event)
            crop_lr_end_event = augment(crop_lr_end_event)
            crop_lr_total_event = augment(crop_lr_total_event)

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples_nb(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            # 'gt' : crop_hr[1],
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'lr_gt': crop_lr[1]
        }

@register('sr-liif-test-temp-kd-nb-downsampled')
class SRliiftestDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = [data['start'], data['gt'], data['end']]
        start_event = data['start_event']
        end_event = data['end_event']
        total_event = data['total_event']

        w_hr = 1280
        h_hr = 720
        w_lr = 320
        h_lr = 180
        x0 = (img[0].shape[-2] - h_hr)//2
        y0 = (img[0].shape[-1] - w_hr)//2

        crop_hr = []
        crop_lr = []
        for i in range(len(img)):
            crop_hr.append(img[i][:, x0: x0 + h_hr, y0: y0 + w_hr])
            crop_lr.append(resize_fn(crop_hr[i], (h_lr, w_lr)))
        crop_hr_start_event = start_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_start_event = resize_event(crop_hr_start_event, (h_lr, w_lr))
        crop_hr_end_event = end_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_end_event = resize_event(crop_hr_end_event, (h_lr, w_lr))
        crop_hr_total_event = total_event[:, x0: x0 + h_hr, y0: y0 + w_hr]
        crop_lr_total_event = resize_event(crop_hr_total_event, (h_lr, w_lr))

        time = (data['frameRange'][1] - data['frameRange'][0])
        time /= (data['frameRange'][2] - data['frameRange'][0])

        hr_coord, hr_rgb = to_pixel_samples_nb(crop_hr[1].contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr[1].shape[-2]
        cell[:, 1] *= 2 / crop_hr[1].shape[-1]

        return {
            'start' : crop_lr[0],
            'end' : crop_lr[2],
            'start_event' : crop_lr_start_event,
            'end_event' : crop_lr_end_event,
            'total_event' : crop_lr_total_event,
            'time' : time,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'lr_gt': crop_lr[1]
        }

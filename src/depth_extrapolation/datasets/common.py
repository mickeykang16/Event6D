import random
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import pyramid_gaussian
from skimage.measure import block_reduce

import torch

def _apply(func, x):

    if isinstance(x, (list, tuple)):
        return [_apply(func, x_i) for x_i in x]
    elif isinstance(x, dict):
        y = {}
        for key, value in x.items():
            y[key] = _apply(func, value)
        return y
    else:
        return func(x)

def crop(args, ps, py, px):    # patch_size
    # args = [input, target]
    def _get_shape(*args):
        if isinstance(args[0], (list, tuple)):
            return _get_shape(args[0][0])
        elif isinstance(args[0], dict):
            return _get_shape(list(args[0].values())[0])
        else:
            return args[0].shape

    h, w, _ = _get_shape(args)
    # print(_get_shape(args))
    # print(ps[1])

    # py = random.randrange(0, h-ps+1)
    # px = random.randrange(0, w-ps+1)

    def _crop(img):
        if img.ndim == 2:
            return img[py:py+ps[1], px:px+ps[0], np.newaxis]
        else:
            return img[py:py+ps[1], px:px+ps[0], :]

    return _apply(_crop, args)

def crop_with_event(*args, left_event, right_event, ps=256):    # patch_size
    # args = [input, target]
    def _get_shape(*args):
        if isinstance(args[0], (list, tuple)):
            return _get_shape(args[0][0])
        elif isinstance(args[0], dict):
            return _get_shape(list(args[0].values())[0])
        else:
            return args[0].shape

    h, w, _ = _get_shape(args)

    py = random.randrange(0, h-ps+1)
    px = random.randrange(0, w-ps+1)

    def _crop(img):
        if img.ndim == 2:
            return img[py:py+ps, px:px+ps, np.newaxis]
        else:
            return img[py:py+ps, px:px+ps, :]

    def _event_crop(event):
        return event[:, py:py+ps, px:px+ps]

    return _apply(_crop, args), _apply(_event_crop, left_event), _apply(_event_crop, right_event)

def crop_event(args, ps, py, px):    # patch_size
    # args = [input, target]
    def _get_shape(*args):
        if isinstance(args[0], (list, tuple)):
            return _get_shape(args[0][0])
        elif isinstance(args[0], dict):
            return _get_shape(list(args[0].values())[0])
        else:
            return args[0].shape

    _, h, w = _get_shape(args)

    # py = random.randrange(0, h-ps+1)
    # px = random.randrange(0, w-ps+1)

    def _crop(img):
        if img.ndim == 2:
            return img[py:py+ps, px:px+ps, np.newaxis]
        else:
            return img[py:py+ps, px:px+ps, :]

    def _event_crop(event):
        return event[:, py:py+ps[1], px:px+ps[0]]

    return _apply(_event_crop, args)

def crop_disp(args, ps, py, px):    # patch_size
    # args = [input, target]
    def _get_shape(*args):
        if isinstance(args[0], (list, tuple)):
            return _get_shape(args[0][0])
        elif isinstance(args[0], dict):
            return _get_shape(list(args[0].values())[0])
        else:
            return args[0].shape

    h, w = _get_shape(args)

    # py = random.randrange(0, h-ps+1)
    # px = random.randrange(0, w-ps+1)

    def _crop(img):
        if img.ndim == 2:
            return img[py:py+ps[1], px:px+ps[0], np.newaxis]
        else:
            return img[py:py+ps[1], px:px+ps[0], :]

    def _disp_crop(event):
        return event[py:py+ps[1], px:px+ps[0]]

    return _apply(_disp_crop, args)

def add_noise(*args, sigma_sigma=2, rgb_range=255):

    if len(args) == 1:  # usually there is only a single input
        args = args[0]

    sigma = np.random.normal() * sigma_sigma * rgb_range/255

    def _add_noise(img):
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        return (img + noise).clip(0, rgb_range)

    return _apply(_add_noise, args)

def augment(*args, hflip=True, rot=True, shuffle=True, change_saturation=True, rgb_range=255):
    """augmentation consistent to input and target"""

    choices = (False, True)

    hflip = hflip and random.choice(choices)
    vflip = rot and random.choice(choices)
    rot90 = rot and random.choice(choices)
    # shuffle = shuffle

    if shuffle:
        rgb_order = list(range(3))
        random.shuffle(rgb_order)
        if rgb_order == list(range(3)):
            shuffle = False

    if change_saturation:
        amp_factor = np.random.uniform(0.5, 1.5)

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        if shuffle and img.ndim > 2:
            if img.shape[-1] == 3:  # RGB image only
                img = img[..., rgb_order]

        if change_saturation:
            hsv_img = rgb2hsv(img)
            hsv_img[..., 1] *= amp_factor

            img = hsv2rgb(hsv_img).clip(0, 1) * rgb_range

        return img.astype(np.float32)

    return _apply(_augment, args)

def pad(img, divisor=4, pad_width=None, negative=False):

    def _pad_numpy(img, divisor=4, pad_width=None, negative=False):
        if pad_width is None:
            (h, w, _) = img.shape
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))

        img = np.pad(img, pad_width, mode='edge')

        return img, pad_width

    def _pad_tensor(img, divisor=4, pad_width=None, negative=False):

        n, c, h, w = img.shape
        if pad_width is None:
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = (0, pad_w, 0, pad_h)
        else:
            try:
                pad_h = pad_width[0][1]
                pad_w = pad_width[1][1]
                if isinstance(pad_h, torch.Tensor):
                    pad_h = pad_h.item()
                if isinstance(pad_w, torch.Tensor):
                    pad_w = pad_w.item()

                pad_width = (0, pad_w, 0, pad_h)
            except:
                pass

            if negative:
                pad_width = [-val for val in pad_width]

        img = torch.nn.functional.pad(img, pad_width, 'reflect')

        return img, pad_width

    if isinstance(img, np.ndarray):
        return _pad_numpy(img, divisor, pad_width, negative)
    else:   # torch.Tensor
        return _pad_tensor(img, divisor, pad_width, negative)

def disp_pad(img, divisor=4, pad_width=None, negative=False):

    def _pad_numpy(img, divisor=4, pad_width=None, negative=False):
        if pad_width is None:
            (h, w, _) = img.shape
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))

        img = np.pad(img, pad_width, mode='edge')

        return img, pad_width

    def _pad_tensor(img, divisor=4, pad_width=None, negative=False):

        if isinstance(img, list):
            img = img[0].unsqueeze(1)
            n, c, h, w = img.shape
        else:
            img = img.unsqueeze(1)
            n, c, h, w = img.shape
        if pad_width is None:
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = (0, pad_w, 0, pad_h)

        else:
            try:
                pad_h = pad_width[0][1][0]
                pad_w = pad_width[1][1][0]
                if isinstance(pad_h, torch.Tensor):
                    pad_h = pad_h.item()
                if isinstance(pad_w, torch.Tensor):
                    pad_w = pad_w.item()

                pad_width = (0, pad_w, 0, pad_h)
            except:
                pass

            if negative:
                pad_width = [-val for val in pad_width]

        img = torch.nn.functional.pad(img, pad_width, 'reflect')

        return img

    if isinstance(img, np.ndarray):
        return _pad_numpy(img, divisor, pad_width, negative)
    else:   # torch.Tensor
        return _pad_tensor(img, divisor, pad_width, negative)

def event_pad(img, divisor=4, pad_width=None, negative=False):

    def _pad_numpy(img, divisor=4, pad_width=None, negative=False):

        if pad_width is None:
            (_, h, w) = img.shape
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = ((0, 0), (0, pad_h), (0, pad_w))

        img = np.pad(img, pad_width, mode='edge')

        return img, pad_width

    def _pad_tensor(img, divisor=4, pad_width=None, negative=False):

        n, c, h, w = img.shape
        if pad_width is None:
            pad_h = -h % divisor
            pad_w = -w % divisor
            pad_width = (0, pad_w, 0, pad_h)
        else:
            try:
                pad_h = pad_width[0][1]
                pad_w = pad_width[1][1]
                if isinstance(pad_h, torch.Tensor):
                    pad_h = pad_h.item()
                if isinstance(pad_w, torch.Tensor):
                    pad_w = pad_w.item()

                pad_width = (0, pad_w, 0, pad_h)
            except:
                pass

            if negative:
                pad_width = [-val for val in pad_width]

        img = torch.nn.functional.pad(img, pad_width, 'reflect')

        return img, pad_width

    if isinstance(img, np.ndarray):
        return _pad_numpy(img, divisor, pad_width, negative)
    else:   # torch.Tensor
        return _pad_tensor(img, divisor, pad_width, negative)

def generate_pyramid(*args, n_scales):

    def _generate_pyramid(img):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        pyramid = list(pyramid_gaussian(img, n_scales-1, multichannel=True))

        return pyramid

    return _apply(_generate_pyramid, args)

def generate_event_pyramid(*args, n_scales):
    def _generate_event_pyramid(event):
        event = np.array(event)

        if event.dtype != np.float32:
            event = event.astype(np.float32)

        # print(event.shape)
        pyramid = []
        for i in range(n_scales):
            w, h, c = event.shape
            # scale_event = block_reduce(event, (w//(2**i) , h//(2**i), c), np.max)
            scale_event = block_reduce(event, (1, 2**i, 2**i), np.max)
            # print(scale_event)
            # print(event)
            pyramid.append(scale_event)

        # pyramid = list(pyramid_gaussian(img, n_scales-1, multichannel=True))
        return pyramid
    return _generate_event_pyramid(args)

def np2tensor(*args):
    def _np2tensor(x):
        np_transpose = np.ascontiguousarray(x.transpose(2, 0, 1))
        tensor = torch.from_numpy(np_transpose)

        return tensor

    return _apply(_np2tensor, args)

def image2tensor(x):
    np_transpose = np.ascontiguousarray(x.transpose(2, 0, 1))
    tensor = torch.from_numpy(np_transpose)

    return tensor

def event2tensor(*args):
    def _np2tensor(x):
        # np_transpose = np.ascontiguousarray(x.transpose(2, 0, 1))
        tensor = torch.from_numpy(x)
        return tensor

    return _apply(_np2tensor, args)

def to(*args, device=None, dtype=torch.float):

    def _to(x):
        return x.to(device=device, dtype=dtype, non_blocking=True, copy=False)

    return _apply(_to, args)

# src/datasets/basic_transforms.py

import random
import math
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2


def to_tensor(img):
    """
    PIL / ndarray -> torch.FloatTensor [C,H,W]
    """
    return TF.to_tensor(img)


def normalize(img, mean, std):
    """
    img: torch tensor [C,H,W]
    """
    return TF.normalize(img, mean=mean, std=std)

def resize(img, w, h=None):
    """
    Resize to (w,h). If h=None, keep aspect ratio.
    """
    if h is None:
        return TF.resize(img, w)
    return TF.resize(img, (h, w))


def crop_to_square(img):
    """
    Center crop to square with edge=min(w,h)
    """
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def crop(img, w, h, c=None):
    """
    Crop WxH around center c=(cx,cy). If c=None, use image center.
    """
    W, H = img.size
    if c is None:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = c

    left = int(cx - w // 2)
    top = int(cy - h // 2)
    return img.crop((left, top, left + w, top + h))


def random_crop(img, w, h):
    """
    Random crop WxH
    """
    W, H = img.size
    if W < w or H < h:
        raise ValueError("Crop size larger than image")

    left = random.randint(0, W - w)
    top = random.randint(0, H - h)
    return img.crop((left, top, left + w, top + h))

def shift(img, dx, dy):
    """
    Shift image by (dx,dy) pixels.
    """
    return TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0)


def random_shift(img, max_dx, max_dy):
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    return shift(img, dx, dy)

def _dynamic_resize(img, angle):
    """
    Resize image so rotation does not introduce black borders.
    """
    w, h = img.size
    angle = math.radians(abs(angle))
    new_w = int(abs(w * math.cos(angle)) + abs(h * math.sin(angle)))
    new_h = int(abs(h * math.cos(angle)) + abs(w * math.sin(angle)))
    return TF.resize(img, (new_h, new_w))


def rotation(img, angle, dynamic_resize=True):
    """
    Fixed rotation.
    """
    print(type(img))
    w, h = img.size
    if dynamic_resize:
        img = _dynamic_resize(img, angle)
        img = TF.rotate(img, angle)
        return crop(img, w, h)
    else:
        return TF.rotate(img, angle)


def random_rotation(img, min=None, max=None, list=None, dynamic_resize=True):
    """
    Supports:
      - continuous range [min,max]
      - discrete list
    """
    if list is not None:
        rot = random.choice(list)
    elif min is not None and max is not None:
        rot = random.uniform(min, max)
    else:
        raise ValueError("Provide either list or min/max")

    return rotation(img, rot, dynamic_resize)

def random_h_flip(img, prob=0.5):
    if random.random() < prob:
        return TF.hflip(img)
    return img

def h_flip(img):
    return TF.hflip(img)


def random_v_flip(img, prob=0.5):
    if random.random() < prob:
        return TF.vflip(img)
    return img

def set_brightness(img, value):
    return TF.adjust_brightness(img, value)


def random_brightness(img, min=0.8, max=1.2):
    return TF.adjust_brightness(img, random.uniform(min, max))

def color_jitter(
    image,
    flip_prob=0.5,
    brightness=0.2,
    contrast=0.2,
    saturation=0.1,
    hue=0.05,
    blur_prob=0.2,
    noise_prob=0.2,
    jitter_p=0.5,
    sharp_lightness=(0.7, 1.0),
    sharp_alpha=(0.2, 0.4),
    sharp_prob=0.5,
):
    """
    Supports both fixed and random behavior depending on args.
    """
    # Albumentations post-processing
    post_aug = A.Compose([
        A.HorizontalFlip(p=flip_prob),
        A.VerticalFlip(p=flip_prob),
        A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=jitter_p
        ),
        A.Sharpen(
            alpha=sharp_alpha,
            lightness=sharp_lightness,
            p=sharp_prob
        ),
        A.GaussianBlur(blur_limit=(3, 3), p=blur_prob),
        A.GaussNoise(var_limit=(0, 10.0), p=noise_prob),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    image = np.array(image)
    return post_aug(image=image)["image"]

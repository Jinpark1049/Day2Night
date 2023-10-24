import os
import torch
import numpy as np
import random
from PIL import Image


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)

def denormalize_image(image, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]):
    """
    Returns Denormalized images

    - Args
        image (torch.Tensor): an image tensor with (N,C,H,W) or (C,H,W)
        mean (List[float]): a list of mean values of each channels used in a normalization (ImageNet)
        std (List[float]): a list of standard deviations used in a normalization (ImageNet)
    """
    
    mean = 255.0*np.array(mean).reshape(-1,1,1)
    std = 255.0*np.array(std).reshape(-1,1,1)

    if len(image.shape) == 4 and image.shape[0]==1:
        image = image.squeeze()
    
    denorm_image = np.clip(image*std+mean, 0, 255)

    return denorm_image

def save_imgs(save_dir, file_name, img):
    img = Image.fromarray(img.astype(np.uint8))

    img.save(os.path.join(save_dir, str(file_name)+'.png'))

import random
import numpy as np
from PIL import Image

def to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def random_crop_pair(rain, gt, crop_size: int):
    h, w = rain.shape[:2]
    if h < crop_size or w < crop_size:
        return rain, gt
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    return rain[y:y+crop_size, x:x+crop_size], gt[y:y+crop_size, x:x+crop_size]

def hflip_pair(rain, gt, p=0.5):
    if random.random() < p:
        return rain[:, ::-1].copy(), gt[:, ::-1].copy()
    return rain, gt

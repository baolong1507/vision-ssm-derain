import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(img_size: int, crop_size: int, is_train: bool):
    common = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_REFLECT),
    ]
    if is_train:
        aug = common + [
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5,
                               p=0.5, border_mode=cv2.BORDER_REFLECT),
        ]
    else:
        aug = common + [
            A.CenterCrop(crop_size, crop_size),
        ]

    aug += [
        A.Normalize((0.5,)*3, (0.5,)*3),
        ToTensorV2(),
    ]

    return A.Compose(
        aug,
        additional_targets={"image_gt": "image"}  # target xử lý như image
    )

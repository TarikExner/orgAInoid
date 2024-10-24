import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class CustomIntensityAdjustment(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CustomIntensityAdjustment, self).__init__(always_apply = always_apply, p = p)
        self.adjustment = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit = (-0.5, 0.5),
                    contrast_limit = (-0.5, 0.5),
                    p=1
                ),
                A.RandomGamma(
                    gamma_limit = (60, 140),
                    p=1
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                #A.GaussNoise(var_limit=(1,1), p=0.5),
                A.AdvancedBlur(p=1)
            ], p=1.0)
        ])

    def apply(self, img, **params):
        non_zero_mask = img > 0
        img_augmented = self.adjustment(image=img)["image"]
        
        img = np.where(non_zero_mask, img_augmented, img)
        
        img_min = img.min()
        img_max = img.max()
        
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        return img

def val_transformations() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),
        ToTensorV2()
    ])

def to_normalized_tensor() -> A.Compose:
    return val_transformations()


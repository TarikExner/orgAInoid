import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
import numpy as np

class CustomIntensityAdjustment(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CustomIntensityAdjustment, self).__init__(always_apply = always_apply, p = p)
        self.adjustment = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit = (-0.2, 0.2),
                    contrast_limit = (-0.2, 0.2),
                    p=1
                ),
                A.RandomGamma(
                    gamma_limit = (60, 140),
                    p=1
                ),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=0.5
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
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

    def get_transform_init_args_names(self):
        return []

class NormalizeSegmented(DualTransform):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=False, p=1.0):
        super(NormalizeSegmented, self).__init__(always_apply, p)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def apply(self, img, **params):
        mask = params['mask']

        mask = mask.astype(bool)

        img_normalized = np.copy(img).astype(np.float32)
        for c in range(3):  # Loop over channels
            img_normalized[..., c][mask[..., c]] = (
                img[..., c][mask[..., c]] - self.mean[c]
            ) / self.std[c]

        return img_normalized.astype(np.float32)

    def apply_to_mask(self, img, **params):
        return img

    def get_transform_init_args_names(self):
        return ("mean", "std")

    @property
    def targets_as_params(self):
        return ['mask']
    
    def get_params_dependent_on_targets(self, params):
        return {'mask': params['mask']}

def val_transformations() -> A.Compose:
    return A.Compose([
        NormalizeSegmented(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def to_normalized_tensor() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


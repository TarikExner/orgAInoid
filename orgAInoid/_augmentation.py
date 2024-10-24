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
        
        # img_min = img.min()
        # img_max = img.max()
        
        # if img_max > img_min:
        #     img = (img - img_min) / (img_max - img_min)
        
        return img


class NormalizeSegmented(DualTransform):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=False, p=1.0):
        super(NormalizeSegmented, self).__init__(always_apply, p)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def apply(self, img, mask=None, **params):
        if mask is None:
            raise ValueError("Mask is required for NormalizeSegmented transformation.")

        # Ensure the mask is correctly shaped to match the image dimensions
        mask = mask.astype(bool)

        # Get the pixels that are not zero using the mask
        non_zero_pixels = img[mask == 0]

        # Calculate mean and std only on non-zero pixels
        mean = non_zero_pixels.mean(axis=0)
        std = non_zero_pixels.std(axis=0)

        # Normalize only non-zero pixels using the pre-defined mean and std
        img_normalized = np.copy(img).astype(np.float32)
        img_normalized[mask == 0] = (non_zero_pixels - mean) / std
        img_normalized = img_normalized * self.std + self.mean

        return img_normalized

    def apply_to_mask(self, mask, **params):
        return mask

    def get_transform_init_args_names(self):
        return ("mean", "std")


def val_transformations() -> A.Compose:
    return A.Compose([
        NormalizeSegmented(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def to_normalized_tensor() -> A.Compose:
    return val_transformations()


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

        # assert np.min(img) == 0
        # assert np.max(img) == 1
        
        return img

    def get_transform_init_args_names(self):
        return []
    


# class NormalizeSegmented(DualTransform):
#     def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=False, p=1.0):
#         super(NormalizeSegmented, self).__init__(always_apply, p)
#         self.mean = np.array(mean)
#         self.std = np.array(std)
# 
#     def apply(self, img, **params):
#         mask = params['mask']
# 
#         if np.max(img) != 1.0 or np.min(img) != 0.0:
#             img = (img - np.min(img)) / (np.max(img)-np.min(img))
# 
#         # Ensure the mask is correctly shaped to match the image dimensions
#         mask = mask.astype(bool)
# 
#         # Get the pixels that are not zero using the mask
#         non_zero_pixels = img[mask == 0]
# 
#         # Calculate mean and std only on non-zero pixels
#         mean = non_zero_pixels.mean(axis=0)
#         std = non_zero_pixels.std(axis=0)
# 
#         # Calculate fill_value using (0-mean)/std
#         fill_value = (0 - mean) / std
# 
#         # Normalize only non-zero pixels using the pre-defined mean and std
#         img_normalized = np.copy(img).astype(np.float32)
#         img_normalized[mask == 0] = (non_zero_pixels - mean) / std
# 
#         # Identify new zero-pixels introduced by augmentations (which were not part of the original mask)
#         new_zero_pixels = (img == 0) & (mask != 0)
# 
#         # Set these newly introduced zero-pixels to the calculated fill_value
#         img_normalized[new_zero_pixels] = fill_value
# 
#         # Rescale to the final mean and std
#         img_normalized = img_normalized * self.std + self.mean
# 
#         return img_normalized.astype(np.float32)
# 
#     def apply_to_mask(self, img, **params):
#         return img
# 
#     def get_transform_init_args_names(self):
#         return ("mean", "std")
# 
#     @property
#     def targets_as_params(self):
#         return ['mask']
#     
#     def get_params_dependent_on_targets(self, params):
#         return {'mask' : params['mask']}

class NormalizeSegmented(DualTransform):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=False, p=1.0):
        super(NormalizeSegmented, self).__init__(always_apply, p)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def apply(self, img, **params):
        mask = params['mask']

        # Ensure mask is boolean and matches image dimensions
        mask = mask.astype(bool)

        # Normalize the non-zero pixels in the ROI
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


import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def val_transformations() -> A.Compose:
    return A.Compose([

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])


class CustomIntensityAdjustment(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CustomIntensityAdjustment, self).__init__(always_apply = always_apply, p = p)
        self.adjustment = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),  # Random brightness/contrast
                A.RandomGamma(p=0.5),               # Random gamma adjustment
            ], p=1.0)
        ])

    def apply(self, img, **params):
        # Apply intensity changes only to non-zero pixels
        non_zero_mask = img > 0
        img_augmented = self.adjustment(image=img)["image"]
        
        # Only change the intensity of non-zero pixels
        img = np.where(non_zero_mask, img_augmented, img)
        
        # Rescale the entire image to the 0-1 range
        img_min = img.min()
        img_max = img.max()
        
        if img_max > img_min:  # To avoid division by zero
            img = (img - img_min) / (img_max - img_min)
        
        return img


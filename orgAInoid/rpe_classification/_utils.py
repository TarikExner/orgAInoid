from torch.utils.data import Dataset
import numpy as np
import torch


class ClassificationDataset(Dataset):
    def __init__(self, image_arr: np.ndarray, classes: np.ndarray, transforms):
        self.image_arr = image_arr
        self.classes = classes
        self.transforms = transforms
    
    def __len__(self):
        return self.image_arr.shape[0]
    
    def __getitem__(self, idx):

        image = self.image_arr[idx, :, :, :]

        corr_class = torch.tensor(self.classes[idx])

        if self.transforms is not None:
            image = self.transforms(image)

        assert isinstance(image, torch.Tensor)
        assert not torch.isnan(image).any()
        assert not torch.isinf(image).any()

        return (image, corr_class)


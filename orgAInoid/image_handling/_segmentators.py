import torch
import numpy as np
import os
from abc import abstractmethod

from typing import Union

from ..segmentation import UNet, DEEPLABV3, HRNET
from .._augmentation import to_normalized_tensor


class MaskPredictor:
    def __init__(self,
                 input_dir: str,
                 input_size: int,
                 model: Union[HRNET, DEEPLABV3, UNet]) -> None:
        self._input_dir = input_dir
        self._input_size = input_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self._load_weights()

    def _load_weights(self):
        state_dict_path = os.path.join(
            self._input_dir,
            f"{self.model.__class__.__name__}_{self._input_size}_bs16.pth"
        )
        self.model.load_state_dict(
            torch.load(
                state_dict_path,
                map_location=torch.device(self.device)
            )
        )
        self.model.eval()
        self.model.to(self.device)

    def _predict_mask_from_model(self,
                                 img_tensor: torch.Tensor) -> torch.Tensor:
        """\
        Creates the mask for an image.

        img_tensor
            The preprocessed image as a tensor. 

        Returns
        -------
        The raw mask without any thresholding.

        """
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            pred_mask = self.model(img_tensor)
            pred_mask = torch.sigmoid(pred_mask)
        return pred_mask

    @abstractmethod
    def _preprocess_image(self,
                          img: np.ndarray) -> torch.Tensor:
        pass

    def _to_normalized_tensor(self,
                              img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = np.expand_dims(img, axis = 2)
        transforms = to_normalized_tensor()
        transformed_image = transforms(image = img)["image"]
        assert isinstance(transformed_image, torch.Tensor), type(transformed_image)
        return transformed_image

    def mask_image(self,
                   img: np.ndarray) -> np.ndarray:
        assert hasattr(self, "_preprocess_image")
        img_tensor = self._preprocess_image(img)
        mask_tensor = self._predict_mask_from_model(img_tensor)
        return mask_tensor.squeeze().detach().cpu().numpy()

    def _add_dim_to_tensor(self,
                           input_tensor):
        return input_tensor.unsqueeze(0)


class UNetPredictor(MaskPredictor):
    """\
    Class to handle segmentation of the images using UNet.
    """

    def __init__(self,
                 input_dir: str,
                 input_size: int) -> None:

        model = UNet()
        super().__init__(input_dir = input_dir,
                         input_size = input_size,
                         model = model)

    def _preprocess_image(self,
                          img: np.ndarray) -> torch.Tensor:
        """Function to preprocess the image for UNet segmentation"""
        transformed_image = self._to_normalized_tensor(img)
        transformed_image = self._add_dim_to_tensor(transformed_image)
        return transformed_image

class DeepLabPredictor(MaskPredictor):
    """\
    Class to handle segmentation of the images using DeepLabV3.
    """

    def __init__(self,
                 input_dir: str,
                 input_size: int) -> None:

        model = DEEPLABV3()
        super().__init__(input_dir = input_dir,
                         input_size = input_size,
                         model = model)

    def _preprocess_image(self,
                          img: np.ndarray) -> torch.Tensor:
        """Function to preprocess the image for UNet segmentation"""
        transformed_image = self._to_normalized_tensor(img)
        transformed_image = transformed_image.repeat(3,1,1)
        transformed_image = self._add_dim_to_tensor(transformed_image)
        return transformed_image

class HRNETPredictor(MaskPredictor):
    """\
    Class to handle segmentation of the images using DeepLabV3.
    """

    def __init__(self,
                 input_dir: str,
                 input_size: int) -> None:

        model = HRNET()
        super().__init__(input_dir = input_dir,
                         input_size = input_size,
                         model = model)

    def _preprocess_image(self,
                          img: np.ndarray) -> torch.Tensor:
        """Function to preprocess the image for UNet segmentation"""
        transformed_image = self._to_normalized_tensor(img)
        transformed_image = transformed_image.repeat(3,1,1)
        transformed_image = self._add_dim_to_tensor(transformed_image)
        return transformed_image


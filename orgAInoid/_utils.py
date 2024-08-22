from .segmentation.model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import cv2
import torch
from scipy.ndimage import zoom
import skimage
import os
from os import PathLike

from typing import Optional


def _generate_file_table(experiment_id: str,
                         image_dir: str,
                         annotations_file: str):
    annotations = pd.read_csv(annotations_file)
    annotations["well"] = [
        entry.split(experiment_id)[1]
        if not entry == f"{experiment_id}{experiment_id}" else experiment_id
        for entry in annotations["ID"].tolist()
    ]
    
    file_names = []
    wells = []
    positions = []
    loops = []
    slices = []
    rpe_annotations = []
    experiments = []
    files = os.listdir(image_dir)
    files = [file for file in files if file.endswith(".tif")]
    for file_name in files:
        
        contents = file_name.split("-")
        contents = [entry for entry in contents if entry != ""]
        if len(contents) != 14:
            print(f"Invalid image: {file_name}")
            continue
        file_names.append(file_name)
        wells.append(contents[0])
        positions.append(contents[1])
        loops.append(contents[2])
        slices.append(contents[4])
        rpe_annotations.append(annotations.loc[annotations["well"] == contents[0], "RPE"].iloc[0])
        experiments.append(experiment_id)
    
    file_dict = {}
    file_dict["experiment"] = experiments
    file_dict["file_name"] = file_names
    file_dict["well"] = wells
    file_dict["position"] = positions
    file_dict["loop"] = loops
    file_dict["slice"] = slices
    file_dict["RPE"] = rpe_annotations
    
    return pd.DataFrame(file_dict)
    

class UNetPredictor:
    """\
    Class to handle segmentation of the images.
    The class expects pre-processed images and will
    only return the pure mask.
    """

    def __init__(self,
                 unet_input_dir: str,
                 unet_input_size: float) -> None:
        self._segmentator_input_dir = unet_input_dir
        self._input_size = unet_input_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_unet_model()

    def _load_unet_model(self) -> UNet:

        state_dict_path = os.path.join(
            self._segmentator_input_dir,
            f"segmentator_{self._input_size}_bs64.pth"
        )
        
        model = UNet(1)
        model.load_state_dict(
            torch.load(
                state_dict_path,
                map_location=torch.device(self.device)
            )
        )
        model.eval()
        model.to(self.device)
        return model

    def create_mask(self,
                    image: np.ndarray) -> np.ndarray:
        """\
        Creates the mask for an image.

        img
            The preprocessed image. For our UNET segmentation,
            that means a normalized and scaled image

        Returns
        -------
        The mask without thresholding.
        """
        image_dimension = image.shape[0]

        assert image_dimension == self._input_size

        transforms = A.Compose([
            A.Resize(image_dimension, image_dimension),
            ToTensorV2()
        ])
        preprocessed = transforms(image = image)
        img_tensor = preprocessed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_mask = self.model(
                img_tensor.reshape(1,1,image_dimension,image_dimension)
            ).squeeze()
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask.cpu().detach().numpy()
        return pred_mask


class OrganoidImage:
    """\
    Class to represent an image and its corresponding methods.
    """

    def __init__(self,
                 path: str):
        self.img = self._read_image(path)
        

    def _read_image(self,
                    path: PathLike) -> np.ndarray:
        """Reads an image from disk and returns it with its original bitdepth"""
        img = cv2.imread(path, -1)
        self.bitdepth = 8 * img.itemsize
        return img.astype(np.float32)

    def _normalize_image(self,
                         img: np.ndarray,
                         bitdepth: int) -> np.ndarray:
        """Applies bit-depth normalization"""
        return img / float(2**bitdepth)

    def _min_max_scale(self,
                       img: np.ndarray,
                       mask_array: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies MinMaxScaling"""
        if mask_array is None:
            min_val = img.min()
            max_val = img.max()
            return (img - min_val) / (max_val - min_val)
        else:
            mask_array = mask_array.astype(bool)
            non_masked_pixels = img[mask_array]
            min_val = non_masked_pixels.min()
            max_val = non_masked_pixels.max()
            img_rescaled = img.copy()
            img_rescaled[mask_array] = (img_rescaled[mask_array] - min_val) / (max_val - min_val)
            return img_rescaled

    def preprocess_for_unet(self,
                            unet_input_size: int) -> None:
        """Applies downsampling, bitdepth normalization and MinMaxScaling"""
        img = self.img.copy()

        # TODO: replace _bin_image with downsample for faster conversion

        # binning_factor = int(img.shape[0] / unet_input_size)
        # img = self._bin_image(img, binning_factor)
        img = self._downsample_for_unet(img, unet_input_size)
        img = self._normalize_image(img, self.bitdepth)
        img = self._min_max_scale(img)
        self.unet_preprocessed = img

    def _downsample_for_unet(self,
                             img: np.ndarray,
                             unet_input_size: int) -> np.ndarray:
        return cv2.resize(img, (unet_input_size, unet_input_size), interpolation=cv2.INTER_NEAREST)


    def downsample(self,
                   target_size: int):
        if isinstance(self, OrganoidMask):

            self.threshold_mask()
        else:
            self.img = cv2.resize(self.img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
    def _bin_image(self,
                   img: np.ndarray,
                   bin_size: int) -> np.ndarray:
        """
        Bins the image on demand.
        """
        height, width = img.shape[:2]

        new_height = height // bin_size
        new_width = width // bin_size

        if new_height == height and new_width == width:
            return img
        
        binned_image = np.zeros((new_height, new_width), dtype=np.float32)

        for i in range(new_height):
            for j in range(new_width):
                binned_image[i, j] = np.mean(img[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size])

        return binned_image


class OrganoidMask(OrganoidImage):
    """\
    Class to represent a mask. This mask is by definition binary.

    Expects a raw mask with gray values between 0 and 1 that will
    be thresholded upon initiation.

    The methods are shared between OrganoidImage and OrganoidMask.
    """

    def __init__(self,
                 raw_mask: np.ndarray,
                 threshold: float = 0.5):
        self.threshold = threshold
        self.img = raw_mask
        self.threshold_mask()
        self.error_while_cleaning = False

    def threshold_mask(self) -> None:
        """Thresholds a non-binary image into a binary"""
        self.img[self.img >= self.threshold] = 1
        self.img[self.img < self.threshold] = 0

    def polynomial_upsample(self,
                            target_size: float,
                            order: int = 1) -> None:
        current_size = self.img.shape[0]
        upsample_factor = int(target_size / current_size)
        self.img = zoom(self.img, upsample_factor, order=order)
        self.threshold_mask()

    def clean_mask(self,
                   min_size_perc: float = 5):
        min_size = int(self.img.shape[0]**2 * (min_size_perc / 100))
        label_objects, num_labels = skimage.measure.label(
            self.img,
            background = 0,
            return_num = True
        )
        if num_labels > 1:
            mask = skimage.morphology.remove_small_objects(
                label_objects, min_size=min_size
            ).astype(np.float32)

            label_objects, num_labels = skimage.measure.label(
                mask,
                background = 0,
                return_num = True
            )

            if num_labels == 0:
                print("Removed only label.")
                self.error_while_cleaning = True
            elif num_labels > 1:
                print("More than one object left after removing small objects.")
                self.error_while_cleaning = True
        
        self.img = label_objects.astype(np.uint8)

    def label_mask(self) -> np.ndarray:
        label_objects = skimage.measure.label(self.img, background=0)
        return label_objects



class ImageHandler:
    """\
    Class to handle the image processing.

    For classification, we need masked images.
    For the transformers, we need timeseries of masked images.

    The class contains public convenience methods that handle
    everything for the user. The individual steps can be accessed
    via the private methods.

    """

    def __init__(self,
                 target_image_size: float,
                 unet_input_dir: Optional[str],
                 unet_input_size: Optional[int] = 128):

        if unet_input_dir is not None and unet_input_size is not None:
            self.unet_input_size = unet_input_size
            self.unet: UNetPredictor = UNetPredictor(unet_input_dir, unet_input_size)
        self.target_size = target_image_size

    def crop_to_mask_bounding_box(self,
                                  mask: OrganoidMask,
                                  img: OrganoidImage,
                                  rescale=True,
                                  crop_size=None) -> tuple[OrganoidImage, OrganoidMask]:
        """
        Processes the image and mask based on the given parameters.
        
        If `flexible` is True, it calculates the bounding box of the ROI, makes it square,
        adds 10 pixels padding to each side, and resizes the image to 224x224.
        
        If `flexible` is False, it asserts that `crop_size` is provided, pads the bounding box
        to match the specified `crop_size`, and resizes the image to 224x224.
        
        Parameters:
        - mask: OrganoidMask, an object containing the binary mask image.
        - img: Image object, the corresponding image to be processed.
        - flexible: bool, whether to use the flexible padding method or the fixed crop size method.
        - crop_size: int, the desired size of the square crop if `flexible` is False.
        
        Returns:
        - img: Processed image object with the resized image.
        - mask: Processed mask with the resized mask.
        """
        
        # Find the coordinates of the bounding box of the ROI in the mask
        rows = np.any(mask.img, axis=1)
        cols = np.any(mask.img, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Calculate the width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        if rescale:
            # Rescale mode: Pad 10 pixels to each side and rescale
            
            # Make the bounding box square by expanding the shorter dimension
            if width > height:
                pad = (width - height) // 2
                y_min = max(0, y_min - pad)
                y_max = min(mask.img.shape[0], y_max + pad)
            else:
                pad = (height - width) // 2
                x_min = max(0, x_min - pad)
                x_max = min(mask.img.shape[1], x_max + pad)

            # Add 10 pixels in each direction, while keeping within image bounds
            x_min = max(0, x_min - 10)
            x_max = min(mask.img.shape[1], x_max + 10)
            y_min = max(0, y_min - 10)
            y_max = min(mask.img.shape[0], y_max + 10)

            # Crop the image and the mask to the bounding box
            img_cropped = img.img[y_min:y_max, x_min:x_max]
            mask_cropped = mask.img[y_min:y_max, x_min:x_max]

        else:
            # Fixed crop size mode: Assert crop_size is provided and pad to match crop_size
            assert crop_size is not None, "crop_size must be provided when flexible is False"
            
            # Calculate padding required to reach the desired crop size
            pad_x = max(0, (crop_size - width) // 2)
            pad_y = max(0, (crop_size - height) // 2)

            # Ensure the padding does not exceed the image boundaries
            x_min = max(0, x_min - pad_x)
            x_max = min(mask.img.shape[1], x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(mask.img.shape[0], y_max + pad_y)

            # If the cropped area is smaller than the desired size, pad the difference
            top_pad = (crop_size - (y_max - y_min)) // 2
            bottom_pad = crop_size - (y_max - y_min) - top_pad
            left_pad = (crop_size - (x_max - x_min)) // 2
            right_pad = crop_size - (x_max - x_min) - left_pad

            # Apply padding to make the cropped area the desired size
            img_cropped = img.img[y_min:y_max, x_min:x_max].copy()
            mask_cropped = mask.img[y_min:y_max, x_min:x_max].copy()
            
            padding_kwargs = {
                "top": top_pad,
                "bottom": bottom_pad,
                "left": left_pad,
                "right": right_pad,
                "borderType": cv2.BORDER_CONSTANT,
                "value": [0]
            }
            img_padded = cv2.copyMakeBorder(
                img_cropped,
                **padding_kwargs
            )
            mask_padded = cv2.copyMakeBorder(
                mask_cropped,
                **padding_kwargs
            )

            # Update img_cropped and mask_cropped with padded versions
            img_cropped, mask_cropped = img_padded, mask_padded
        

        # Resize the cropped image to 224x224 using appropriate interpolation
        img_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_cropped, (224, 224), interpolation=cv2.INTER_NEAREST)

        # Update the original image and mask
        img.img = img_resized
        mask.img = mask_resized

        return img, mask

    def create_mask_from_image(self,
                               img: OrganoidImage,
                               clean: bool = True,
                               min_size_perc: float = 7.5) -> OrganoidMask:

        img.preprocess_for_unet(self.unet_input_size)
        
        raw_mask = self.unet.create_mask(img.unet_preprocessed)
        mask = OrganoidMask(raw_mask)

        if mask.img.shape[0] < self.target_size:
            mask.polynomial_upsample(self.target_size)
        else:
            mask.downsample(self.target_size)

        if clean:
            mask.clean_mask(min_size_perc)

        assert mask.img.shape == (self.target_size, self.target_size)
        try:
            assert np.max(mask.img) == 1
        except AssertionError:
            assert mask.error_while_cleaning is True
        assert np.min(mask.img) == 0

        return mask

    def get_masked_image(self,
                         img: OrganoidImage,
                         crop_bounding_box: bool = True,
                         rescale: bool = True,
                         crop_size: Optional[int] = None,
                         normalized: bool = False,
                         scaled: bool = True) -> Optional[OrganoidImage]:

        mask: OrganoidMask = self.create_mask_from_image(img,
                                                         clean = True)

        if mask.error_while_cleaning:
            print("Masking went wrong... Returning None")
            return None
        
        img.downsample(self.target_size)

        assert mask.img.shape == img.img.shape
        
        if crop_bounding_box:
            img, mask = self.crop_to_mask_bounding_box(mask, img, rescale = rescale, crop_size = crop_size)

        img.img = img.img * mask.img

        if normalized:
            img.img = img._normalize_image(img.img, bitdepth = 16)
        if scaled:
            img.img = img._min_max_scale(img.img, mask_array = mask.img)
        
        return img

    def read_image(self,
                   path) -> OrganoidImage:
        return OrganoidImage(path)








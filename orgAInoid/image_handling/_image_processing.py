import numpy as np
import cv2

from typing import Optional

from ._images import OrganoidImage, OrganoidMask

class ImageProcessor:
    """\
    Class to handle image processing.
    """

    def __init__(self):
        pass

    def _threshold_mask(self,
                        img: np.ndarray,
                        threshold: float,
                        copy: bool = False) -> np.ndarray:
        img = img.copy() if copy else img
        img[img >= threshold] = 1
        img[img < threshold] = 0
        return img

    def threshold_mask(self,
                       img: np.ndarray,
                       threshold: float,
                       copy: bool = False) -> np.ndarray:
        return self._threshold_mask(img,
                                    threshold,
                                    copy)

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

    def preprocess_for_segmentation(self,
                                    img: np.ndarray,
                                    segmentator_input_size: int,
                                    copy: bool = True) -> np.ndarray:
        """Applies downsampling, bitdepth normalization and MinMaxScaling"""
        img = img.copy() if copy else img
        img = self._downsample(img, segmentator_input_size)
        img = self._min_max_scale(img)
        return img

    def _already_has_right_size(self,
                                img: np.ndarray,
                                size: int):
        return img.shape == (size, size)

    def _downsample(self,
                    img: np.ndarray,
                    target_size: int) -> np.ndarray:
        if self._already_has_right_size(img, target_size):
            return img
        return cv2.resize(
            img,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA
        )

    def downsample(self,
                   img: np.ndarray,
                   target_size: int) -> np.ndarray:
        return self._downsample(img, target_size)

    def _upsample(self,
                  img: np.ndarray,
                  target_size: int) -> np.ndarray:
        if self._already_has_right_size(img, target_size):
            return img
        return cv2.resize(
            img,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR
        )

    def upsample(self,
                 img: np.ndarray,
                 target_size: int) -> np.ndarray:
        return self._upsample(img, target_size)

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

        if not mask.is_binary():
            thresholded_mask = self._threshold_mask(mask.image, threshold = 0.5, copy = True)
            rows = np.any(thresholded_mask, axis=1)
            cols = np.any(thresholded_mask, axis=0)
        else:
            rows = np.any(mask.image, axis=1)
            cols = np.any(mask.image, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        width = x_max - x_min
        height = y_max - y_min

        if rescale:
            if width > height:
                pad = (width - height) // 2
                y_min = max(0, y_min - pad)
                y_max = min(mask.image.shape[0], y_max + pad)
            else:
                pad = (height - width) // 2
                x_min = max(0, x_min - pad)
                x_max = min(mask.image.shape[1], x_max + pad)

            x_min = max(0, x_min - 10)
            x_max = min(mask.image.shape[1], x_max + 10)
            y_min = max(0, y_min - 10)
            y_max = min(mask.image.shape[0], y_max + 10)

            img_cropped = img.image[y_min:y_max, x_min:x_max]
            mask_cropped = mask.image[y_min:y_max, x_min:x_max]

        else:
            assert crop_size is not None, "crop_size must be provided when flexible is False"
            
            pad_x = max(0, (crop_size - width) // 2)
            pad_y = max(0, (crop_size - height) // 2)

            x_min = max(0, x_min - pad_x)
            x_max = min(mask.image.shape[1], x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(mask.image.shape[0], y_max + pad_y)

            top_pad = (crop_size - (y_max - y_min)) // 2
            bottom_pad = crop_size - (y_max - y_min) - top_pad
            left_pad = (crop_size - (x_max - x_min)) // 2
            right_pad = crop_size - (x_max - x_min) - left_pad

            img_cropped = img.image[y_min:y_max, x_min:x_max].copy()
            mask_cropped = mask.image[y_min:y_max, x_min:x_max].copy()
            
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

            img_cropped, mask_cropped = img_padded, mask_padded
        
        img.image = img_cropped
        mask.image = mask_cropped

        return img, mask

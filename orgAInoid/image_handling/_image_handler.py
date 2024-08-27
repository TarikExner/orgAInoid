from typing import Optional, Literal
import numpy as np

from . import (
    OrganoidImage, ImageProcessor,
    HRNETPredictor, DeepLabPredictor, UNetPredictor,
    MaskPredictor, OrganoidMask
)


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
                 segmentator_input_dir: str,
                 segmentator_input_size: int,
                 segmentation_model_name: Literal["HRNET", "UNET", "DEEPLABV3"] = "DEEPLABV3"):

        self.img_processor = ImageProcessor()
        self.segmentator = self._initialize_segmentation_model(
            segmentation_model_name,
            input_dir = segmentator_input_dir,
            input_size = segmentator_input_size
        )
        self.segmentator_input_size = segmentator_input_size

    def get_mask_and_image(self,
                           img: OrganoidImage,
                           image_target_dimension: int,
                           mask_threshold: float = 0.5,
                           clean_mask: bool = True,
                           min_size_percentage: float = 7.5,
                           crop_bounding_box: bool = True,
                           rescale_cropped_image: bool = False,
                           crop_bounding_box_dimension: int = 320) -> tuple[Optional[np.ndarray],...]:

        # First, we create the mask
        mask: OrganoidMask = self._create_mask_from_image(img)

        if crop_bounding_box is True:
            try:
                img, mask = self.img_processor.crop_to_mask_bounding_box(
                    mask,
                    img,
                    rescale = rescale_cropped_image,
                    crop_size = crop_bounding_box_dimension
                )
                assert not mask.is_binary()
            except Exception:
                print("Cropping failed!")
                return None, None

        if mask.dimension != image_target_dimension:
            if mask.dimension < image_target_dimension:
                mask.image = self.img_processor.upsample(mask.image, image_target_dimension)
            else:
                mask.image = self.img_processor.downsample(mask.image, image_target_dimension)

        mask.image = self.img_processor.threshold_mask(mask.image, mask_threshold)

        if clean_mask is True:
            try:
                mask.clean_mask(min_size_percentage)
                assert mask.is_binary()

            except ValueError as e:
                print(str(e))
                return None, None

        if img.dimension != image_target_dimension:
            if img.dimension < image_target_dimension:
                img.image = self.img_processor.upsample(img.image, image_target_dimension)
            else:
                img.image = self.img_processor.downsample(img.image, image_target_dimension)
        
        assert img.shape == mask.shape

        return img.image, mask.image

    def get_masked_image(self,
                         img: OrganoidImage,
                         image_target_dimension: int,
                         mask_threshold: float = 0.5,
                         clean_mask: bool = True,
                         min_size_percentage: float = 7.5,
                         scale_masked_image: bool = True,
                         crop_bounding_box: bool = True,
                         rescale_cropped_image: bool = False,
                         crop_bounding_box_dimension: Optional[int] = 320) -> Optional[np.ndarray]:

        """\
        Method to get a masked image as a numpy array.
        Will downsample the image, if necessary and apply a suitable
        mask to it. The mask will be cleaned automatically.

        Ideally, we keep the mask in an unthresholded state as long as possible
        """

        unmasked_image, final_mask = self.get_mask_and_image(
            img = img,
            image_target_dimension = image_target_dimension,
            mask_threshold = mask_threshold,
            clean_mask = clean_mask,
            min_size_percentage = min_size_percentage,
            crop_bounding_box = crop_bounding_box,
            rescale_cropped_image = rescale_cropped_image,
            crop_bounding_box_dimension = crop_bounding_box_dimension
        )

        if unmasked_image is None or final_mask is None:
            return None

        masked = unmasked_image * final_mask

        if scale_masked_image:
            masked = self.img_processor._min_max_scale(masked, mask_array = final_mask)
            assert np.max(masked) == 1
            assert np.min(masked) == 0

        return masked

    def _create_mask_from_image(self,
                                img: OrganoidImage) -> OrganoidMask:
        """\
        Takes an OrganoidImage and calculates the mask. The shape is
        determined by the self.segmentator_input_size parameter. No
        rescaling or similar operations are performed. The raw mask
        without thresholding is returned.
        """
        img_array = img.image
        preprocessed = self.img_processor.preprocess_for_segmentation(
            img_array,
            self.segmentator_input_size
        )
        assert preprocessed.shape == (self.segmentator_input_size, self.segmentator_input_size)
        return OrganoidMask(self.segmentator.mask_image(preprocessed))

        


    def _initialize_segmentation_model(self,
                                       model_name: Literal["HRNET", "UNET", "DEEPLABV3"],
                                       input_dir: str,
                                       input_size: int) -> MaskPredictor:
        kwargs = {
            "input_dir": input_dir,
            "input_size": input_size
        }
        if model_name == "UNET":
            return UNetPredictor(**kwargs)
        elif model_name == "DEEPLABV3":
            return DeepLabPredictor(**kwargs)
        elif model_name == "HRNET":
            return HRNETPredictor(**kwargs)
        else:
            raise ValueError(f"Unknown model name {model_name}!")


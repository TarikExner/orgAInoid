import numpy as np
import cv2

import skimage

from typing import Optional

class OrganoidImage:
    """\
    Class to represent an image and its corresponding methods.
    """

    def __init__(self,
                 path: Optional[str]):
        if path is None:
            self._img = np.zeros(shape = (0,0))
        else:
            self._img = self._read_image(path)

    def _read_image(self,
                    path: str) -> np.ndarray:
        """\
        Reads an image from disk and returns it with its original bitdepth.
        We make sure that we actually directly return it as 32bit float.
        """
        img = cv2.imread(path, -1) # -1 for original bitdepth
        return img.astype(np.float32)

    @property
    def image(self) -> np.ndarray:
        return self._img

    @image.setter
    def image(self,
              img: np.ndarray):
        self._img = img

    @property
    def shape(self):
        return self.image.shape

    @property
    def dimension(self):
        assert self.image.shape[0] == self.image.shape[1]
        return self.image.shape[0]


class OrganoidMask(OrganoidImage):
    """\
    Class to represent a mask. Raw masks can be any gray value
    and are not necessarily binary already!
    """

    def __init__(self,
                 raw_mask: np.ndarray):
        super().__init__(path = None)
        self._img = raw_mask

    def is_clean(self,
                 thresholded_image: np.ndarray) -> bool:
        """Checks if there is only one label after thresholding"""
        assert self._is_binary(thresholded_image)
        _, num_labels = self.label_mask_with_counts(thresholded_image)
        return num_labels == 1

    def clean_mask(self,
                   min_size_perc: float):

        assert self.is_binary()

        min_size = int(self.dimension**2 * (min_size_perc / 100))

        labeled_mask, num_labels = self.label_mask_with_counts(self._img)

        if num_labels > 1:
            mask = skimage.morphology.remove_small_objects(
                labeled_mask, min_size=min_size
            ).astype(np.float32)

            labeled_mask, num_labels = self.label_mask_with_counts(mask)

            if num_labels == 0:
                raise ValueError("Removed only Label")
            elif num_labels > 1:
                raise ValueError("More than one object left after removing small objects.")

        self.image = labeled_mask.astype(np.uint8)

    def label_mask(self,
                   img_array: np.ndarray) -> np.ndarray:
        labels = skimage.measure.label(
            img_array,
            background = 0
        )
        assert isinstance(labels, np.ndarray)
        return labels

    def label_mask_with_counts(self,
                               img_array: np.ndarray) -> tuple[np.ndarray, int]:
        labels, num_labels = skimage.measure.label(
            img_array,
            background = 0,
            return_num = True
        )
        assert isinstance(labels, np.ndarray)
        assert isinstance(num_labels, int)
        return labels, num_labels

    def is_binary(self):
        return self._is_binary(self._img)

    def _is_binary(self,
                   img: np.ndarray):
        return np.isin(img, [0, 1]).all()


class OrganoidMaskImage(OrganoidImage):
    """\
    Class to represent a mask image. This mask is by definition binary.
    """

    def __init__(self,
                 path: str) -> None:
        """\
        Reads an image and directly converts it to the correct binary scale"""
        super().__init__(path)
        if np.max(self._img) != 1:
            self._img /= np.max(self._img)
        assert self.is_binary()

    def is_binary(self):
        return np.isin(self._img, [0, 1]).all()

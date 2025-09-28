from skimage.filters import laplace
from scipy.stats import mode, skew
import numpy as np
from scipy.spatial import ConvexHull

########################################
# Parameters we can calculate directly #
########################################


def aspect_ratio(regionprops_table: dict):
    major_axis_length = regionprops_table["axis_major_length"][0]
    minor_axis_length = regionprops_table["axis_minor_length"][0]
    return np.array([major_axis_length / minor_axis_length])


def roundness(regionprops_table: dict):
    area = regionprops_table["area"][0]
    major_axis_length = regionprops_table["axis_major_length"][0]
    return np.array([(4 * area) / (np.pi * (major_axis_length**2))])


def compactness(regionprops_table: dict):
    area = regionprops_table["area"][0]
    perimeter = regionprops_table["perimeter"][0]
    return np.array([(perimeter**2) / (4 * np.pi * area)])


def circularity(regionprops_table: dict):
    area = regionprops_table["area"][0]
    perimeter = regionprops_table["perimeter"][0]
    return np.array([(4 * np.pi * area) / (perimeter**2)])


def form_factor(regionprops_table: dict):
    area = regionprops_table["area"][0]
    perimeter = regionprops_table["perimeter"][0]
    return np.array([4 * np.pi * area / np.sqrt(perimeter)])


def effective_diameter(regionprops_table: dict):
    area = regionprops_table["area"][0]
    return np.array([np.sqrt(area / np.pi) * 2])


def convexity(regionprops_table: dict):
    coords = regionprops_table["coords"][0]
    hull = ConvexHull(coords)
    hull_perimeter = np.sum(
        np.sqrt(np.sum(np.diff(coords[hull.vertices], axis=0) ** 2, axis=1))
    )
    hull_perimeter += np.sqrt(
        np.sum((coords[hull.vertices[0]] - coords[hull.vertices[-1]]) ** 2)
    )
    perimeter = regionprops_table["perimeter"][0]
    return np.array([hull_perimeter / perimeter])


##############################################
# Parameters we can calculate from the image #
# and have to supply via extra_properties    #
##############################################


def blur(mask, img):
    mask = mask.astype(bool)
    return np.var(laplace(img[mask]))


def roi_contrast(mask, img):
    mask = mask.astype(bool)
    return np.max(img[mask]) - np.min(img[mask])


def image_contrast(mask, img):
    mask = mask.astype(bool)
    return np.max(img) - np.min(img)


def intensity_median(mask, img):
    mask = mask.astype(bool)
    return np.median(img[mask])


def modal_value(mask, img):
    mask = mask.astype(bool)
    return np.array([mode(img[mask], axis=None).mode])


def integrated_density(mask, img):
    mask = mask.astype(bool)
    area = np.count_nonzero(mask)
    return area * np.mean(img[mask])


def raw_integrated_density(mask, img):
    mask = mask.astype(bool)
    return np.sum(img[mask])


def skewness(mask, img):
    mask = mask.astype(bool)
    return skew(img[mask])


def kurtosis(mask, img):
    from scipy.stats import kurtosis

    mask = mask.astype(bool)  #
    return kurtosis(img[mask], axis=None)

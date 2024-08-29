from . import methods

EXTRA_PROPERTIES = [
    methods.blur,
    methods.roi_contrast,
    methods.image_contrast,
    methods.intensity_median,
    methods.modal_value,
    methods.integrated_density,
    methods.raw_integrated_density,
    methods.skewness,
    methods.kurtosis
]

PROPERTIES = [
    "label",
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
    "coords",
    "centroid",
    "centroid_local",
    "centroid_weighted",
    "centroid_weighted_local",
    "eccentricity",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "image_convex",
    "image_filled",
    "image_intensity",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
    "moments", # array
    "moments_central", # array
    "moments_hu", # array
    "moments_normalized", # array
    "moments_weighted", # array
    "moments_weighted_central", # array
    "moments_weighted_hu", # tuple
    "moments_weighted_normalized", # array
    "num_pixels",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity"
]

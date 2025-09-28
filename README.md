# Repo for the Organoid Code

## Code repository for organoid tissue prediction

The paper has been published at bioRxiv: https://www.biorxiv.org/content/10.1101/2025.02.19.639061v1

Please cite the article as:

article {Afting2025.02.19.639061,
	author = {Afting, Cassian and Bhatti, Norin and Schlagheck, Christina and Salvador, Encarnaci{\'o}n S{\'a}nchez and Herrera-Astorga, Laura and Agarwal, Rashi and Suzuki, Risa and Hackert, Nicolaj and Lorenz, Hanns-Martin and Zilova, Lucie and Wittbrodt, Joachim and Exner, Tarik},
	title = {Deep learning predicts tissue outcomes in retinal organoids},
	elocation-id = {2025.02.19.639061},
	year = {2025},
	doi = {10.1101/2025.02.19.639061},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/02/23/2025.02.19.639061},
	eprint = {https://www.biorxiv.org/content/early/2025/02/23/2025.02.19.639061.full.pdf},
	journal = {bioRxiv}
}


One important comment: The code refers to the terms "validation" and "test" set other than the nomenclature commonly used within the field.
The paper itself and all figures define the validation set as the set that is derived from a part of the training data not used for training,
while the test set is assembled from completely unrelated data that were acquired in different experiments. Since this was introduced
into the code at an early timepoint, we decided not to change it in the code.

## Repo structure
The repository contains every line of code that was used for the analysis.

### /segmentation
contains the code for the models and training used to segment the organoids. We initially tried multiple models (HRNet, DeepLabV3 and UNET) and stuck with DeepLabV3

### /morphometrics
contains the code to calculate the morphological features from the segmented images. Here, every additional method for image description is included.

### /image_transformation
contains code that was not used in the final manuscript

### /image_handling
contains utility functions and classes to properly segment and scale the images for classification

### /human_evaluation
contains the imageJ macro that human scientists were given in order to analyze and predict the images

### /figures
contains the code necessary to generate the data for the figures as well as the figures itself.
After running the respective analyses, the figure generation is only dependent on this module and
is executable by running generate_final_figures(), which will run the final analyses and figure assembly
as shown in the paper.

### /discover
contains code in order to try to replicate the DISCOVER method (Nature Communications 2023). We did not use it in the end.

### /classification
contains the code for the neural networks, the training loops and the dataset assemblies. Also included is the code for the
classifier experiments with algorithms of "classical" machine learning.

## Use the classification datasets

The datasets have been deposited at Zenodo:
- [Single slice](https://zenodo.org/records/17202714)  
- [SUM projections](https://zenodo.org/records/17205312)  
- [Maximum projections](https://zenodo.org/records/17205321)

In order to use it, you will have to clone the repository.

```python
from orgAInoid.classification import OrganoidDataset
data = OrganoidDataset.read_classification_dataset("./path/to/file.cds")
# metadata are stored at the .metadata attribute
data.metadata # contains a pd.DataFrame that links image information to the array index

# scaled and segmented images are stored under .X
X = data.X # returns a np.ndarray of shape (n_images, 1, 224, 224)

# class annotations are stored in a dictionary under .y[readout]
# where readout is one of ['RPE_Final', 'Lens_Final', 'RPE_classes',
# 'Lens_classes', 'Total_RPE_amount', 'Lens_area', 'morph_classes']
y = data.y["RPE_Final"]

# dataset metadata are stored under .dataset_metadata, containing some information
# about class distribution and the presence/absence of readouts
ds_metadata = data.dataset_metadata

# image metadata are stored under .image_metadata, containing information about
# the image dimensions and segmentation settings
img_metadata = data.image_metadata

# in order to merge data, use the .merge() method
scnd_ds = OrganoidDataset.read_classification_dataset("./path/to/other/file.cds")
data = data.merge(scnd_ds)
# change the dataset ID as you created a new one
data.dataset_metadata.dataset_id = "E001-6_full_SL3_fixed"
# save under a different name, the dataset_id will be used as a file name
data.save("./save/path", overwrite = True)

``` 


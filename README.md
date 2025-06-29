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
	abstract = {Retinal organoids have become important models for studying development and disease, yet stochastic heterogeneity in the formation of cell types, tissues, and phenotypes remains a major challenge. This limits our ability to precisely experimentally address the early developmental trajectories towards these outcomes. Here, we utilize deep learning to predict the differentiation path and resulting tissues in retinal organoids well before they become visually discernible. Our approach effectively bypasses the challenge of organoid-related heterogeneity in tissue formation. For this, we acquired a high-resolution time-lapse imaging dataset comprising about 1,000 organoids and over 100,000 images enabling precise temporal tracking of organoid development. By combining expert annotations with advanced image analysis of organoid morphology, we characterized the heterogeneity of the retinal pigmented epithelium (RPE) and lens tissues, as well as global organoid morphologies over time. Using this training set, our deep learning approach accurately predicts the emergence and size of RPE and lens tissue formation on an organoid-by-organoid basis at early developmental stages, refining our understanding of when early lineage decisions are made. This approach advances knowledge of tissue and phenotype decision-making in organoid development and can inform the design of similar predictive platforms for other organoid systems, paving the way for more standardized and reproducible organoid research. Finally, it provides a direct focus on early developmental time points for in-depth molecular analyses, alleviated from confounding effects of heterogeneity.Competing Interest StatementThe authors have declared no competing interest.},
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


import pandas as pd
import torch

from typing import Literal, Optional, Callable

from skimage.segmentation import slic
from ._graph_utils import (get_images_and_masks)
from ._saliency_utils import (get_dataloader,
                              masks_to_tensor,
                              initialize_models,
                              create_baseline_image,
                              compute_integrated_gradients,
                              compute_saliency,
                              compute_gradient_shap,
                              compute_deeplift_shap,
                              compute_grad_cam,
                              compute_guided_grad_cam,
                              compute_smooth_occlusion,
                              compute_feature_ablation,
                              compute_kernel_shap)

from ..classification._dataset import OrganoidDataset
from ..image_handling._image_handler import ImageHandler

Readouts = Literal["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]

SALIENCY_FUNCTIONS: dict[str, Callable] = {
    # gradient based methods
    "IG_NT": compute_integrated_gradients,
    "SAL_NT": compute_saliency,
    "GRS": compute_gradient_shap,
    "DLS": compute_deeplift_shap,
    # gradcam methods
    "GC": compute_grad_cam,
    "GGC": compute_guided_grad_cam,
    # occlusion based methods
    "OCC": compute_smooth_occlusion,
    "FAB": compute_feature_ablation,
    "KSH": compute_kernel_shap
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _find_layer(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer {layer_name} not found!")

def _define_target_layers(model) -> Optional[torch.nn.Module]:
    if model.__class__.__name__ == "DenseNet121":
        return _find_layer(model, "densenet121.features.denseblock1.denselayer1.conv2")
    if model.__class__.__name__ == "MobileNetV3_Large":
        return _find_layer(model, "mobilenet_v3_large.features.1.conv.6")
    if model.__class__.__name__ == "ResNet50":
        return _find_layer(model, "resnet50.layer1.0.conv3")
    raise ValueError(f"Unknown model {model.__class__.__name__}")

def compute_saliencies(dataset: OrganoidDataset,
                       readout: Readouts,
                       model_directory: str,
                       baseline_directory: str,
                       well: Optional[str] = None,
                       cnn_models: list[str] = ["DenseNet121", "ResNet50", "MobileNetV3_Large"],
                       segmentator_input_dir: str = "../segmentation/segmentators") -> None:

    img_handler = ImageHandler(
        segmentator_input_dir = segmentator_input_dir,
        segmentator_input_size = dataset.image_metadata.segmentator_input_size,
        segmentation_model_name = "DEEPLABV3"
    )
    metadata: pd.DataFrame = dataset.metadata

    if well is None:
        organoid_wells = metadata["well"].unique()
    else:
        organoid_wells = [well]

    experiments = metadata["experiment"].unique()
    if len(experiments) > 1:
        raise ValueError("There should only be one experiment")
    experiment = experiments[0]

    models = initialize_models(cnn_models,
                               experiment,
                               readout,
                               model_directory,
                               baseline_directory)


    for well in organoid_wells:
        loops = dataset.metadata.loc[
            dataset.metadata["well"] == well,
            "loop"
        ].sort_values(ascending = True).tolist()

        well_results = {}

        images, masks = get_images_and_masks(dataset, well, img_handler)
        masks = masks_to_tensor(masks)
        cnn_loader = get_dataloader(dataset, well, readout)

        for model_name in cnn_models:
            well_results[model_name] = {}

            trained_model = models[model_name]
            baseline_model = models[f"{model_name}_baseline"]

            for i, ((img, cls), mask, loop) in enumerate(zip(cnn_loader, masks, loops)):

                image = images[i][0]
                slic_mask = mask.detach().cpu().numpy().copy()[0]
                slic_labels = slic(
                    image,
                    mask = slic_mask,
                    n_segments = 50,
                    compactness = 0.1,
                    start_label = 1,
                    channel_axis = None
                )
                mask2d = torch.from_numpy(slic_labels).long()
                mask3 = mask2d.unsqueeze(0).repeat(3,1,1)
                mask3 = mask3.unsqueeze(0).to(DEVICE)

                _image = img
                _class = torch.argmax(cls, dim = 1)
                _mask = mask.unsqueeze(0)
                _baseline = create_baseline_image(_image, _mask, method = "mean")
                _image = _image.to("cuda")
                _class = _class.to("cuda")
                _baseline = _baseline.to("cuda")

                image_results = {}
                image_baseline_results = {}

                for name, fn in SALIENCY_FUNCTIONS.items():
                    kwargs = {
                        "image": _image,
                        "baseline": _baseline,
                        "target": _class,
                        "feature_mask": mask3,
                        "nt_samples": 20,
                        "stdevs": 0.1,
                    }

                    kwargs["model"] = trained_model 
                    kwargs["target_layer"] = _define_target_layers(trained_model)
                    attributions = fn(**kwargs)

                    kwargs["model"] = baseline_model
                    kwargs["target_layer"] = _define_target_layers(baseline_model)
                    baseline_attributions = fn(**kwargs)

                    image_results[name] = attributions
                    image_baseline_results[name] = baseline_attributions


                well_results[model_name][loop]["trained"] = image_results
                well_results[model_name][loop]["baseline"] = image_baseline_results


    return well_results

                    






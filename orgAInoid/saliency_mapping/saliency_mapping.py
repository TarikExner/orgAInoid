import os
import numpy as np
import pandas as pd
import torch
import gc
import pickle
import h5py

from typing import Literal, Optional, Callable

from tqdm import tqdm

import warnings

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
    # "GRS": compute_gradient_shap,
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
        return _find_layer(model, "densenet121.features.denseblock1.denselayer3.conv2")
    if model.__class__.__name__ == "MobileNetV3_Large":
        return _find_layer(model, "mobilenet_v3_large.features.2.block.2.0")
    if model.__class__.__name__ == "ResNet50":
        return _find_layer(model, "resnet50.layer2.0.conv3")
    raise ValueError(f"Unknown model {model.__class__.__name__}")

def save_saliency_h5(all_results: dict,
                     experiment: str,
                     well: str,
                     readout: str,
                     output_dir: str
                     ):
    # { well: { loop: { model: { algorithm: { status: array }}}}}
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{experiment}_{well}_{readout}.h5")

    with h5py.File(file_path, "w") as h5f:
        for well, loops in all_results.items():
            grp_well = h5f.require_group(str(well))

            for loop_idx, models in loops.items():
                grp_loop = grp_well.require_group(str(loop_idx))

                for model_name, algs in models.items():
                    grp_model = grp_loop.require_group(model_name)

                    for alg_name, statuses in algs.items():
                        grp_alg = grp_model.require_group(alg_name)

                        for status, array in statuses.items():  # “trained” or “baseline”
                            # cast to float32
                            arr32 = array.astype(np.float32)

                            # write under: /…/alg_name/status
                            if status in grp_alg:
                                del grp_alg[status]
                            grp_alg.create_dataset(
                                status,
                                data=arr32,
                                compression="gzip",
                                chunks=arr32.shape
                            )

    print(f"Saved saliency maps to {file_path}")

def compute_saliencies(dataset: OrganoidDataset,
                       readout: Readouts,
                       model_directory: str,
                       baseline_directory: str,
                       well: Optional[str] = None,
                       cnn_models: list[str] = ["DenseNet121", "ResNet50", "MobileNetV3_Large"],
                       segmentator_input_dir: str = "../segmentation/segmentators",
                       output_dir: str = "./saliencies",
                       suppress_warnings: bool = True) -> dict:
    if suppress_warnings:
        warnings.filterwarnings("ignore")

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

    target_layers = {
        model: _define_target_layers(models[model])
        for model in models
    }

    all_results = {}

    for well in organoid_wells:
        loops = (
            dataset.metadata
            .loc[dataset.metadata["well"] == well, "loop"]
            .sort_values()
            .tolist()
        )

        images, masks = get_images_and_masks(dataset, well, img_handler)
        masks = masks_to_tensor(masks)

        cnn_loader = get_dataloader(dataset, well, readout)

        well_results = {loop: {} for loop in loops}

        assert len(cnn_loader) == len(loops)

        for i, ((img, cls), mask, loop) in enumerate(
            tqdm(zip(cnn_loader, masks, loops), desc=f"Well {well}", total=len(cnn_loader))
        ):
            image = images[i][0]
            slic_mask = mask.detach().cpu().numpy()[0]
            slic_labels = slic(
                image,
                mask=slic_mask,
                n_segments=50,
                compactness=0.1,
                start_label=1,
                channel_axis=None
            )
            avg_size = np.unique(slic_labels, return_counts=True)[1].mean()
            print("Average segment size: ", avg_size)
            mask2d = torch.from_numpy(slic_labels).long()
            mask3 = mask2d.unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(DEVICE)

            _image = img.to(DEVICE)
            _class = torch.argmax(cls, dim=1).to(DEVICE)
            _baseline = create_baseline_image(img, mask.unsqueeze(0), method="mean").to(DEVICE)

            for model_name in cnn_models:
                trained = models[model_name]
                baseline = models[f"{model_name}_baseline"]
                sample_results = {}

                for fn_name, fn in SALIENCY_FUNCTIONS.items():
                    common_kwargs = {
                        "image": _image,
                        "baseline": _baseline,
                        "target": _class,
                        "feature_mask": mask3,
                        "nt_samples": 20,
                        "stdevs": 0.1,
                    }

                    out_trained = fn(
                        **{**common_kwargs,
                           "model": trained,
                           "target_layer": target_layers[model_name]}
                    )

                    out_base = fn(
                        **{**common_kwargs,
                           "model": baseline,
                           "target_layer": target_layers[f"{model_name}_baseline"]}
                    )
                    out_trained = out_trained.float()
                    out_base = out_base.float()

                    attr_trained = out_trained.detach().cpu().numpy()
                    attr_baseline = out_base.detach().cpu().numpy()

                    sample_results[fn_name] = {
                        "trained": attr_trained,
                        "baseline": attr_baseline 
                    }

                    del out_trained
                    del out_base
                    del attr_trained
                    del attr_baseline

                    # clean up Python-side
                    gc.collect()
                    # free up any cached GPU memory
                    torch.cuda.empty_cache()

                well_results[loop][model_name] = sample_results
                del sample_results
                gc.collect()
                torch.cuda.empty_cache()

        all_results[well] = well_results

    save_saliency_h5(
        all_results = all_results,
        experiment = experiment,
        well = organoid_wells[0],
        readout = readout,
        output_dir = output_dir
    )


    return all_results

                    






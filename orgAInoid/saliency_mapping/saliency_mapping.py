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
    # all_results structure: 
    #   { well_key: { loop_idx: 
    #       "image": np.ndarray,    # shape (1,3,H,W)
    #       "mask":  np.ndarray,    # shape (1,1,H,W)
    #       { model_name: {
    #           alg_name: {"trained": arr2d, "baseline": arr2d}, ...
    #   }}}}

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment}_{well}_{readout}.h5")

    with h5py.File(path, 'w') as h5f:
        for well_key, loops in all_results.items():
            grp_well = h5f.require_group(str(well_key))

            for loop_idx, sample in loops.items():
                grp_loop = grp_well.require_group(str(loop_idx))

                img = sample['image'].astype(np.float32)
                grp_loop.create_dataset(
                    'input_image', data=img,
                    compression="gzip",
                    chunks=img.shape
                )

                msk = sample['mask'].astype(np.float32)
                grp_loop.create_dataset(
                    'mask', data=msk,
                    compression="gzip",
                    chunks=msk.shape
                )

                for model_name, algs in sample.items():
                    if model_name in ('image','mask'):
                        continue     # skip the two ND arrays

                    grp_model = grp_loop.require_group(model_name)
                    for fn_name, statuses in algs.items():
                        grp_fn = grp_model.require_group(fn_name)

                        tr = statuses['trained'].astype(np.float32)
                        grp_fn.create_dataset(
                            'trained', data=tr,
                            compression="gzip",
                            chunks=tr.shape
                        )
                        bl = statuses['baseline'].astype(np.float32)
                        grp_fn.create_dataset(
                            'baseline', data=bl,
                            compression="gzip",
                            chunks=bl.shape
                        )

    print(f"Saved saliency maps to {path}")

def compute_saliencies(dataset: OrganoidDataset,
                       readout: Readouts,
                       model_directory: str,
                       baseline_directory: str,
                       well: Optional[str] = None,
                       combine_images: Literal["mean", "sum"] = "sum",
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

        for i, ((img, _cls), mask, loop) in enumerate(
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
            mask2d = torch.from_numpy(slic_labels).long()
            mask3 = mask2d.unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(DEVICE)

            _image = img.to(DEVICE)
            _class = torch.argmax(_cls, dim=1).to(DEVICE)
            _baseline = create_baseline_image(img, mask.unsqueeze(0), method="mean").to(DEVICE)

            orig_img = img.detach().cpu().numpy().astype(np.float32)

            sample_results = {"image": orig_img, "mask": mask.detach().cpu().numpy()}

            for model_name in cnn_models:
                trained = models[model_name]
                baseline = models[f"{model_name}_baseline"]

                sample_results[model_name] = {}

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

                    attr_trained = out_trained.float().detach().cpu().numpy()
                    attr_baseline = out_base.float().detach().cpu().numpy()

                    if combine_images == "sum":
                        attr_trained = attr_trained[0].sum(axis = 0)
                        attr_baseline = attr_baseline[0].sum(axis = 0)
                    elif combine_images == "mean":
                        attr_trained = attr_trained[0].mean(axis = 0)
                        attr_baseline = attr_baseline[0].mean(axis = 0)
                    else:
                        raise ValueError(f"Unknown combination method {combine_images}")

                    sample_results[model_name][fn_name] = {
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

            well_results[loop] = sample_results
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

                    






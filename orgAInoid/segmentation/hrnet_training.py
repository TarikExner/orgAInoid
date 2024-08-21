from .model import HRNET
from ._utils import _run_segmentation_train_loop


def run_hrnet_training(image_size: int = 2048,
                       batch_size: int = 128,
                       sub_batch_size: int = 4,
                       init_lr = 0.001,
                       score_output_dir: str = "./results",
                       model_output_dir: str = "./segmentators",
                       dataset_dir: str = "./raw_data") -> None:

    model = HRNET()

    _run_segmentation_train_loop(dataset_dir = dataset_dir,
                                 image_size = image_size,
                                 batch_size = batch_size,
                                 model = model,
                                 init_lr = init_lr,
                                 sub_batch_size = sub_batch_size,
                                 score_output_dir = score_output_dir,
                                 model_output_dir = model_output_dir)



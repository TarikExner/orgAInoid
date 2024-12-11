from .model import DEEPLABV3
from ._utils import _run_segmentation_train_loop


def run_deeplabv3_training(image_size: int = 2048,
                           batch_size: int = 128,
                           init_lr = 0.001,
                           n_epochs = 200,
                           score_output_dir: str = "./results",
                           model_output_dir: str = "./segmentators",
                           dataset_dir: str = "./raw_data") -> None:

    model = DEEPLABV3()

    _run_segmentation_train_loop(dataset_dir = dataset_dir,
                                 image_size = image_size,
                                 batch_size = batch_size,
                                 model = model,
                                 n_epochs = n_epochs,
                                 init_lr = init_lr,
                                 score_output_dir = score_output_dir,
                                 model_output_dir = model_output_dir)

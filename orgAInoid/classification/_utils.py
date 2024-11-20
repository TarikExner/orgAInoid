from dataclasses import dataclass, field
from enum import Enum

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional, Union, Literal

from torch_lr_finder import LRFinder

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

from .._augmentation import val_transformations, CustomIntensityAdjustment, NormalizeSegmented

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from abc import abstractmethod
from copy import deepcopy
from sklearn.model_selection._search_successive_halving import _SubsampleMetaSplitter, _top_k
from math import ceil, floor, log
from sklearn.model_selection._search import BaseSearchCV
from numbers import Integral, Real
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.metrics._scorer import get_scorer_names
from sklearn.model_selection._split import _yields_constant_splits, check_cv
from sklearn.base import _fit_context, is_classifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _num_samples

from sklearn.model_selection import ParameterSampler

RPE_CUTOFFS = [1068, 1684]
LENS_CUTOFFS = [17263, 29536]

RPE_MAX = 6500
LENS_MAX = 60_000

RPE_ADJUSTED_CUTOFFS = list(np.array(RPE_CUTOFFS) / RPE_MAX)
LENS_ADJUSTED_CUTOFFS = list(np.array(LENS_CUTOFFS) / LENS_MAX)

class SegmentatorModel(Enum):
    DEEPLABV3 = "DEEPLABV3"
    UNET = "UNET"
    HRNET = "HRNET"


@dataclass
class ImageMetadata:
    dimension: int
    cropped_bbox: bool
    scaled_to_size: bool
    crop_size: Optional[int]
    segmentator_model: SegmentatorModel
    segmentator_input_size: int
    mask_threshold: float
    cleaned_mask: bool
    scale_masked_image: bool
    crop_bounding_box: bool
    rescale_cropped_image: bool
    crop_bounding_box_dimension: Optional[int]

    def __repr__(self):
        return (
            f"ImageMetadata("
            f"dimension={self.dimension}, "
            f"cropped_bbox={self.cropped_bbox}, "
            f"scaled_to_size={self.scaled_to_size}, "
            f"crop_size={self.crop_size}, "
            f"segmentator_model={self.segmentator_model!r}, "
            f"segmentator_input_size={self.segmentator_input_size}, "
            f"mask_threshold={self.mask_threshold}, "
            f"cleaned_mask={self.cleaned_mask}, "
            f"scale_masked_image={self.scale_masked_image}, "
            f"crop_bounding_box={self.crop_bounding_box}, "
            f"rescale_cropped_image={self.rescale_cropped_image}, "
            f"crop_bounding_box_dimension={self.crop_bounding_box_dimension})"
        )

    def __eq__(self, other):
        if not isinstance(other, ImageMetadata):
            return NotImplemented
        return (
            self.dimension == other.dimension and
            self.cropped_bbox == other.cropped_bbox and
            self.scaled_to_size == other.scaled_to_size and
            self.crop_size == other.crop_size and
            self.segmentator_model == other.segmentator_model and
            self.segmentator_input_size == other.segmentator_input_size and
            self.mask_threshold == other.mask_threshold and
            self.cleaned_mask == other.cleaned_mask and
            self.scale_masked_image == other.scale_masked_image and
            self.crop_bounding_box == other.crop_bounding_box and
            self.rescale_cropped_image == other.rescale_cropped_image and
            self.crop_bounding_box_dimension == other.crop_bounding_box_dimension
        )


@dataclass
class DatasetMetadata:
    dataset_id: str
    experiment_dir: str
    readouts: list[str]
    readouts_n_classes: list[int]
    start_timepoint: int
    stop_timepoint: int
    slices: list[str]
    n_slices: int = field(init = False)
    class_balance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_slices = len(self.slices)
        self.timepoints = list(range(self.start_timepoint, self.stop_timepoint + 1))
        self.n_classes_dict = {
            readout: n_class
            for readout, n_class
            in zip(self.readouts, self.readouts_n_classes)
        }

    def calculate_start_and_stop_timepoint(self):
        self.start_timepoint = min(self.timepoints)
        self.stop_timepoint = max(self.timepoints)

    def __repr__(self):
        return (
            f"DatasetMetadata("
            f"dataset_id={self.dataset_id!r}, "
            f"experiment_dir={self.experiment_dir!r}, "
            f"readouts={self.readouts!r}, "
            f"start_timepoint={self.start_timepoint}, "
            f"stop_timepoint={self.stop_timepoint}, "
            f"slices={self.slices!r}, "
            f"n_slices={self.n_slices}, "
            f"class_balance={self.class_balance!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, DatasetMetadata):
            return NotImplemented
        return (
            self.dataset_id == other.dataset_id and
            self.experiment_dir == other.experiment_dir and
            self.readouts == other.readouts and
            self.start_timepoint == other.start_timepoint and
            self.stop_timepoint == other.stop_timepoint and
            self.slices == other.slices and
            self.class_balance == other.class_balance
        )

class ClassificationDataset(Dataset):
    def __init__(self, image_arr: np.ndarray, classes: np.ndarray, transforms):
        self.image_arr = image_arr
        self.classes = classes
        self.transforms = transforms
    
    def __len__(self):
        return self.image_arr.shape[0]
    
    def __getitem__(self, idx):

        image = self.image_arr[idx, :, :, :]
        if image.shape[0] == 1:
            # Duplicate the single channel to create a 3-channel image
            image_3ch = np.repeat(image, 3, axis=0)
        else:
            image_3ch = image

        # Transpose image to [224, 224, 3] for Albumentations
        image_3ch = np.transpose(image_3ch, (1, 2, 0))
        zero_pixel_mask = (image_3ch == 0).astype(np.float32)
        assert image_3ch.shape == zero_pixel_mask.shape

        corr_class = torch.tensor(self.classes[idx])

        if self.transforms is not None:
            augmented = self.transforms(image=image_3ch, mask=zero_pixel_mask)
            image = augmented["image"]

        assert isinstance(image, torch.Tensor)
        assert not torch.isnan(image).any()
        assert not torch.isinf(image).any()

        return (image, corr_class)

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

def find_ideal_learning_rate(model: nn.Module,
                             criterion: nn.Module,
                             optimizer: Optimizer,
                             train_loader: DataLoader,
                             start_lr: Optional[float] = None,
                             end_lr: Optional[float] = None,
                             num_iter: Optional[int] = None,
                             n_tests: int = 5,
                             return_dataframe: bool = False) -> Union[float, pd.DataFrame]:
    start_lr = start_lr or 1e-7
    end_lr = end_lr or 5e-2
    num_iter = num_iter or 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_data = pd.DataFrame()
    for i in range(n_tests):
        lr_finder = LRFinder(model, optimizer, criterion, device = device)
        lr_finder.range_test(train_loader, start_lr = start_lr, end_lr = end_lr, num_iter = num_iter)
        data = pd.DataFrame(lr_finder.history)
        data = data.rename(columns = {"loss": f"run{i}"})
        if i == 0:
            full_data = data
        else:
            full_data = full_data.merge(data, on = "lr")

        lr_finder.reset()
    full_data["mean"] = full_data.groupby(["lr"]).mean().mean(axis = 1).tolist()
    full_data["mean"] = _smooth_curve(full_data["mean"])

    if return_dataframe:
        return full_data

    return _calculate_ideal_learning_rate(full_data)

    # evaluation window is set to 1e-5 to 1e-2 for now

def _calculate_ideal_learning_rate(df: pd.DataFrame):
    
    lr_at_min_loss = df.loc[df["mean"] == df["mean"].min(), "lr"].iloc[0]
    window = df[df["lr"] <= lr_at_min_loss]
    window_start = window.index[0]
    inf_points = _calculate_inflection_points(window["mean"])
    
    # we take the last one which should be closest to the minimum of the fitted function
    inf_point = inf_points[-1] + window_start

    ideal_learning_rate = df.iloc[inf_point]["lr"]

    return ideal_learning_rate

def _smooth_curve(arr, degree: int = 5):
    x = np.arange(arr.shape[0])
    y = arr

    coeff = np.polyfit(x, y, degree)

    polynomial = np.poly1d(coeff)
    return polynomial(x)

def _calculate_inflection_points(y) -> np.ndarray:
    x = np.arange(y.shape[0])
    dy_dx = np.gradient(y,x)
    d2y_dx2 = np.gradient(dy_dx, x)
    inflection_points = np.where(np.diff(np.sign(d2y_dx2)))[0]
    return inflection_points


def train_transformations(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.Rotate(limit=360, p=0.5),  # Random rotation by any angle between -45 and 45 degrees
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=0,  # Set rotate limit to 0 if using Rotate separately
            mask_value = 0,
            p=0.5
        ),  # Shift and scale
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.8, 1),
            p=0.5
        ),  # Resized crop
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            mask_value = 0,
            p=0.5
        ),
        A.Affine(
            scale=1,
            translate_percent=(-0.3, 0.3),
            rotate=0,
            shear=(-15, 15),
            p=0.5
        ),
        # Apply intensity modifications only to non-masked pixels
        CustomIntensityAdjustment(p=0.5),

        A.CoarseDropout(
            max_holes=20,
            min_holes=10,
            max_height=12,
            max_width=12,
            p=0.5
        ),

        # Normalization
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),
        NormalizeSegmented(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Convert to PyTorch tensor
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def create_dataset(img_array: np.ndarray,
                   class_array: np.ndarray,
                   transformations) -> ClassificationDataset:
    return ClassificationDataset(img_array, class_array, transformations)

def create_dataloader(img_array: np.ndarray,
                      class_array: np.ndarray,
                      batch_size: int,
                      shuffle: bool,
                      train: bool,
                      **kwargs) -> DataLoader:
    transformations = train_transformations() if train else val_transformations()
    dataset = create_dataset(img_array, class_array, transformations)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, **kwargs)

def _get_model_from_enum(model_name: str):
    return [
        model for model
        in SegmentatorModel
        if model.value == model_name
    ][0]

class BaseSuccessiveHalving_TE(BaseSearchCV):
    """Implements successive halving.

    Ref:
    Almost optimal exploration in multi-armed bandits, ICML 13
    Zohar Karnin, Tomer Koren, Oren Somekh
    """

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        # overwrite `scoring` since multi-metrics are not supported
        "scoring": [StrOptions(set(get_scorer_names())), callable, None],
        "random_state": ["random_state"],
        "max_resources": [
            Interval(Integral, 0, None, closed="neither"),
            StrOptions({"auto"}),
        ],
        "min_resources": [
            Interval(Integral, 0, None, closed="neither"),
            StrOptions({"exhaust", "smallest"}),
        ],
        "resource": [str],
        "factor": [Interval(Real, 0, None, closed="neither")],
        "aggressive_elimination": ["boolean"],
    }
    _parameter_constraints.pop("pre_dispatch")  # not used in this class

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=5,
        verbose=0,
        random_state=None,
        error_score=np.nan,
        return_train_score=True,
        max_resources="auto",
        min_resources="exhaust",
        resource="n_samples",
        factor=3,
        aggressive_elimination=False,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.random_state = random_state
        self.max_resources = max_resources
        self.resource = resource
        self.factor = factor
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination

    def _check_input_parameters(self, X, y, groups):
        # We need to enforce that successive calls to cv.split() yield the same
        # splits: see https://github.com/scikit-learn/scikit-learn/issues/15149
        if not _yields_constant_splits(self._checked_cv_orig):
            raise ValueError(
                "The cv parameter must yield consistent folds across "
                "calls to split(). Set its random_state to an int, or set "
                "shuffle=False."
            )

        if (
            self.resource != "n_samples"
            and self.resource not in self.estimator.get_params()
        ):
            raise ValueError(
                f"Cannot use resource={self.resource} which is not supported "
                f"by estimator {self.estimator.__class__.__name__}"
            )

        if isinstance(self, HalvingRandomSearchCV_TE):
            if self.min_resources == self.n_candidates == "exhaust":
                # for n_candidates=exhaust to work, we need to know what
                # min_resources is. Similarly min_resources=exhaust needs to
                # know the actual number of candidates.
                raise ValueError(
                    "n_candidates and min_resources cannot be both set to 'exhaust'."
                )

        self.min_resources_ = self.min_resources
        if self.min_resources_ in ("smallest", "exhaust"):
            if self.resource == "n_samples":
                n_splits = self._checked_cv_orig.get_n_splits(X, y, groups)
                # please see https://gph.is/1KjihQe for a justification
                magic_factor = 2
                self.min_resources_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    y = self._validate_data(X="no_validation", y=y, **{"multi_output": True})
                    check_classification_targets(y)
                    n_classes = np.unique(y).shape[0]
                    self.min_resources_ *= n_classes
            else:
                self.min_resources_ = 1
            # if 'exhaust', min_resources_ might be set to a higher value later
            # in _run_search

        self.max_resources_ = self.max_resources
        if self.max_resources_ == "auto":
            if not self.resource == "n_samples":
                raise ValueError(
                    "resource can only be 'n_samples' when max_resources='auto'"
                )
            self.max_resources_ = _num_samples(X)

        if self.min_resources_ > self.max_resources_:
            raise ValueError(
                f"min_resources_={self.min_resources_} is greater "
                f"than max_resources_={self.max_resources_}."
            )

        if self.min_resources_ == 0:
            raise ValueError(
                f"min_resources_={self.min_resources_}: you might have passed "
                "an empty dataset X."
            )

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Custom refit callable to return the index of the best candidate.

        We want the best candidate out of the last iteration. By default
        BaseSearchCV would return the best candidate out of all iterations.

        Currently, we only support for a single metric thus `refit` and
        `refit_metric` are not required.
        """
        last_iter = np.max(results["iter"])
        last_iter_indices = np.flatnonzero(results["iter"] == last_iter)

        test_scores = results["mean_test_score"][last_iter_indices]
        # If all scores are NaNs there is no way to pick between them,
        # so we (arbitrarily) declare the zero'th entry the best one
        if np.isnan(test_scores).all():
            best_idx = 0
        else:
            best_idx = np.nanargmax(test_scores)

        return last_iter_indices[best_idx]

    @_fit_context(
        # Halving*SearchCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_output), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        self._checked_cv_orig = check_cv(
            self.cv, y, classifier=is_classifier(self.estimator)
        )

        self._check_input_parameters(
            X=X,
            y=y,
            groups=groups,
        )

        self._n_samples_orig = _num_samples(X)

        super().fit(X, y=y, groups=groups, **fit_params)

        # Set best_score_: BaseSearchCV does not set it, as refit is a callable
        self.best_score_ = self.cv_results_["mean_test_score"][self.best_index_]

        return self

    def _run_search(self, evaluate_candidates):
        candidate_params = self._generate_candidate_params()

        if self.resource != "n_samples" and any(
            self.resource in candidate for candidate in candidate_params
        ):
            # Can only check this now since we need the candidates list
            raise ValueError(
                f"Cannot use parameter {self.resource} as the resource since "
                "it is part of the searched parameters."
            )

        # n_required_iterations is the number of iterations needed so that the
        # last iterations evaluates less than `factor` candidates.
        n_required_iterations = 1 + floor(log(len(candidate_params), self.factor))

        if self.min_resources == "exhaust":
            # To exhaust the resources, we want to start with the biggest
            # min_resources possible so that the last (required) iteration
            # uses as many resources as possible
            last_iteration = n_required_iterations - 1
            self.min_resources_ = max(
                self.min_resources_,
                self.max_resources_ // self.factor**last_iteration,
            )

        # n_possible_iterations is the number of iterations that we can
        # actually do starting from min_resources and without exceeding
        # max_resources. Depending on max_resources and the number of
        # candidates, this may be higher or smaller than
        # n_required_iterations.
        n_possible_iterations = 1 + floor(
            log(self.max_resources_ // self.min_resources_, self.factor)
        )

        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)

        if self.verbose:
            print(f"n_iterations: {n_iterations}")
            print(f"n_required_iterations: {n_required_iterations}")
            print(f"n_possible_iterations: {n_possible_iterations}")
            print(f"min_resources_: {self.min_resources_}")
            print(f"max_resources_: {self.max_resources_}")
            print(f"aggressive_elimination: {self.aggressive_elimination}")
            print(f"factor: {self.factor}")

        self.n_resources_ = []
        self.n_candidates_ = []

        for itr in range(n_iterations):
            power = itr  # default
            if self.aggressive_elimination:
                # this will set n_resources to the initial value (i.e. the
                # value of n_resources at the first iteration) for as many
                # iterations as needed (while candidates are being
                # eliminated), and then go on as usual.
                power = max(0, itr - n_required_iterations + n_possible_iterations)

            n_resources = int(self.factor**power * self.min_resources_)
            # guard, probably not needed
            n_resources = min(n_resources, self.max_resources_)
            self.n_resources_.append(n_resources)

            n_candidates = len(candidate_params)
            self.n_candidates_.append(n_candidates)

            if self.verbose:
                print("-" * 10)
                print(f"iter: {itr}")
                print(f"n_candidates: {n_candidates}")
                print(f"n_resources: {n_resources}")

            if self.resource == "n_samples":
                # subsampling will be done in cv.split()
                cv = _SubsampleMetaSplitter(
                    base_cv=self._checked_cv_orig,
                    fraction=n_resources / self._n_samples_orig,
                    subsample_test=True,
                    random_state=self.random_state,
                )

            else:
                # Need copy so that the n_resources of next iteration does
                # not overwrite
                candidate_params = [c.copy() for c in candidate_params]
                for candidate in candidate_params:
                    candidate[self.resource] = n_resources
                cv = self._checked_cv_orig

            more_results = {
                "iter": [itr] * n_candidates,
                "n_resources": [n_resources] * n_candidates,
            }

            results = evaluate_candidates(
                candidate_params, cv, more_results=more_results
            )

            n_candidates_to_keep = ceil(n_candidates / self.factor)
            candidate_params = _top_k(results, n_candidates_to_keep, itr)

        self.n_remaining_candidates_ = len(candidate_params)
        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations

    @abstractmethod
    def _generate_candidate_params(self):
        pass

    def _more_tags(self):
        tags = deepcopy(super()._more_tags())
        tags["_xfail_checks"].update(
            {
                "check_fit2d_1sample": (
                    "Fail during parameter check since min/max resources requires"
                    " more samples"
                ),
            }
        )
        return tags

class HalvingRandomSearchCV_TE(BaseSuccessiveHalving_TE):
    """Randomized search on hyper parameters.

    The search strategy starts evaluating all the candidates with a small
    amount of resources and iteratively selects the best candidates, using more
    and more resources.

    The candidates are sampled at random from the parameter space and the
    number of sampled candidates is determined by ``n_candidates``.

    Read more in the :ref:`User guide<successive_halving_user_guide>`.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_halving_search_cv``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_halving_search_cv # noqa
        >>> # now you can import normally from model_selection
        >>> from sklearn.model_selection import HalvingRandomSearchCV

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_candidates : "exhaust" or int, default="exhaust"
        The number of candidate parameters to sample, at the first
        iteration. Using 'exhaust' will sample enough candidates so that the
        last iteration uses as many resources as possible, based on
        `min_resources`, `max_resources` and `factor`. In this case,
        `min_resources` cannot be 'exhaust'.

    factor : int or float, default=3
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for each subsequent iteration. For example,
        ``factor=3`` means that only one third of the candidates are selected.

    resource : ``'n_samples'`` or str, default='n_samples'
        Defines the resource that increases with each iteration. By default,
        the resource is the number of samples. It can also be set to any
        parameter of the base estimator that accepts positive integer
        values, e.g. 'n_iterations' or 'n_estimators' for a gradient
        boosting estimator. In this case ``max_resources`` cannot be 'auto'
        and must be set explicitly.

    max_resources : int, default='auto'
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. By default, this is set ``n_samples`` when
        ``resource='n_samples'`` (default), else an error is raised.

    min_resources : {'exhaust', 'smallest'} or int, default='smallest'
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Equivalently, this defines the amount of
        resources `r0` that are allocated for each candidate at the first
        iteration.

        - 'smallest' is a heuristic that sets `r0` to a small value:

            - ``n_splits * 2`` when ``resource='n_samples'`` for a regression
              problem
            - ``n_classes * n_splits * 2`` when ``resource='n_samples'`` for a
              classification problem
            - ``1`` when ``resource != 'n_samples'``

        - 'exhaust' will set `r0` such that the **last** iteration uses as
          much resources as possible. Namely, the last iteration will use the
          highest value smaller than ``max_resources`` that is a multiple of
          both ``min_resources`` and ``factor``. In general, using 'exhaust'
          leads to a more accurate estimator, but is slightly more time
          consuming. 'exhaust' isn't available when `n_candidates='exhaust'`.

        Note that the amount of resources used at each iteration is always a
        multiple of ``min_resources``.

    aggressive_elimination : bool, default=False
        This is only relevant in cases where there isn't enough resources to
        reduce the remaining candidates to at most `factor` after the last
        iteration. If ``True``, then the search process will 'replay' the
        first iteration for as long as needed until the number of candidates
        is small enough. This is ``False`` by default, which means that the
        last iteration may evaluate more than ``factor`` candidates. See
        :ref:`aggressive_elimination` for more details.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. note::
            Due to implementation details, the folds produced by `cv` must be
            the same across multiple calls to `cv.split()`. For
            built-in `scikit-learn` iterators, this can be achieved by
            deactivating shuffling (`shuffle=False`), or by setting the
            `cv`'s `random_state` parameter to an integer.

    scoring : str, callable, or None, default=None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``HalvingRandomSearchCV`` instance.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for subsampling the dataset
        when `resources != 'n_samples'`. Also used for random uniform
        sampling from lists of possible values instead of scipy.stats
        distributions.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration. It corresponds to `ceil(n_candidates[-1] / factor)`

    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``min_resources_``, the actual
        number of resources used at the last iteration may be smaller than
        ``max_resources_``.

    min_resources_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        The number of iterations that are possible starting with
        ``min_resources_`` resources and without exceeding
        ``max_resources_``.

    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``factor`` candidates at the last iteration, starting with
        ``min_resources_`` resources. This will be smaller than
        ``n_possible_iterations_`` when there isn't enough resources.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``. It contains lots of information
        for analysing the results of a search.
        Please refer to the :ref:`User guide<successive_halving_cv_results>`
        for details.

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    :class:`HalvingGridSearchCV`:
        Search over a grid of parameters using successive halving.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    All parameter combinations scored with a NaN will share the lowest rank.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.experimental import enable_halving_search_cv  # noqa
    >>> from sklearn.model_selection import HalvingRandomSearchCV
    >>> from scipy.stats import randint
    >>> import numpy as np
    ...
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(random_state=0)
    >>> np.random.seed(0)
    ...
    >>> param_distributions = {"max_depth": [3, None],
    ...                        "min_samples_split": randint(2, 11)}
    >>> search = HalvingRandomSearchCV(clf, param_distributions,
    ...                                resource='n_estimators',
    ...                                max_resources=10,
    ...                                random_state=0).fit(X, y)
    >>> search.best_params_  # doctest: +SKIP
    {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 9}
    """

    _required_parameters = ["estimator", "param_distributions"]
    from sklearn.utils._param_validation import Interval, StrOptions
    from numbers import Integral, Real

    _parameter_constraints: dict = {
        **BaseSuccessiveHalving_TE._parameter_constraints,
        "param_distributions": [dict],
        "n_candidates": [
            Interval(Integral, 0, None, closed="neither"),
            StrOptions({"exhaust"}),
        ],
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_candidates="exhaust",
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="smallest",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates

    def _generate_candidate_params(self):
        n_candidates_first_iter = self.n_candidates
        if n_candidates_first_iter == "exhaust":
            # This will generate enough candidate so that the last iteration
            # uses as much resources as possible
            n_candidates_first_iter = self.max_resources_ // self.min_resources_
        return ParameterSampler(
            self.param_distributions,
            n_candidates_first_iter,
            random_state=self.random_state,
        )


def conduct_hyperparameter_search(model,
                                  grid: dict,
                                  method: Literal["HalvingRandomSearchCV",
                                                  "HalvingGridSearchCV",
                                                  "RandomizedSearchCV",
                                                  "GridSearchCV"],
                                  X_train: np.ndarray,
                                  y_train: np.ndarray) -> Union[RandomizedSearchCV, GridSearchCV]:
    n_jobs = 16
    if method == "RandomizedSearchCV":
        total_params = sum(len(grid[key]) for key in grid)
        grid_result = RandomizedSearchCV(estimator = model,
                                         param_distributions = grid,
                                         scoring = "f1_weighted",
                                         n_iter = min(total_params, 20),
                                         verbose = 3,
                                         n_jobs = 8,
                                         cv = 5,
                                         error_score = 0.0,
                                         random_state = 187).fit(X_train, y_train)
    elif method == "GridSearchCV":
        grid_result = GridSearchCV(estimator = model,
                                   param_grid = grid,
                                   scoring = "f1_weighted",
                                   cv = 5,
                                   n_jobs = 8,
                                   verbose = 3,
                                   error_score = 0.0).fit(X_train, y_train)
        
    elif method == "HalvingGridSearchCV":
        raise NotImplementedError("Needs a seperate class")
        #grid_result = HalvingGridSearchCV(estimator = model,
        #                                  param_grid = grid,
        #                                  scoring = "f1_macro",
        #                                  factor = 3,
        #                                  resource = "n_samples",
        #                                  min_resources = 1000,
        #                                  cv = 5,
        #                                  n_jobs = -1,
        #                                  verbose = 3,
        #                                  error_score = 0.0,
        #                                  random_state = 187).fit(X_train, y_train)

    elif method == "HalvingRandomSearchCV":
        grid_result = HalvingRandomSearchCV_TE(estimator = model,
                                               param_distributions = grid,
                                               scoring = "f1_weighted",
                                               factor = 3,
                                               resource = "n_samples",
                                               min_resources = 1000,
                                               cv = 5,
                                               n_jobs = n_jobs,
                                               verbose = 3,
                                               error_score = 0.0,
                                               random_state = 187).fit(X_train, y_train)    

    return grid_result


def _one_hot_encode_labels(labels_array: np.ndarray,
                           readout: str) -> np.ndarray:
    n_classes_dict = {
        "RPE_Final": 2,
        "Lens_Final": 2,
        "RPE_classes": 4,
        "Lens_classes": 4
    }
    n_classes = n_classes_dict[readout]
    n_appended = 0
    if np.unique(labels_array).shape[0] != n_classes:
        # we have not enough labels. That means we look up how many
        # classes there are and provide the according array.

        # first, we provide every item there is potentially as an array
        if "classes" in readout:
            full_class_spectrum = np.array(list(range(4)))
        else:
            full_class_spectrum = np.array(["no", "yes"])
        n_appended = full_class_spectrum.shape[0]

        labels_array = np.hstack([labels_array, full_class_spectrum])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels_array)
    
    onehot_encoder = OneHotEncoder()
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    classification = onehot_encoder.fit_transform(integer_encoded).toarray()

    if n_appended != 0:
        classification = classification[:-n_appended]

    return classification

def _filter_wells(df: pd.DataFrame,
                  combinations: np.ndarray,
                  columns: list[str]):
    combinations_df = pd.DataFrame(combinations, columns=columns)
    return df.merge(combinations_df, on=columns, how='inner')

def _apply_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_to_match = ["experiment", "well"]
    unique_well_per_exp = df[cols_to_match].drop_duplicates().reset_index(drop = True).to_numpy()
    train_wells, test_wells = train_test_split(unique_well_per_exp, test_size = 0.1, random_state = 187)

    assert isinstance(train_wells, np.ndarray)
    assert isinstance(test_wells, np.ndarray)

    train_df = _filter_wells(df, train_wells, ["experiment", "well"])
    test_df = _filter_wells(df, test_wells, ["experiment", "well"])

    return train_df, test_df



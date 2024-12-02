import pickle
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from ._dataset import OrganoidDataset, OrganoidTrainingDataset
from ._utils import create_dataloader, _apply_train_test_split, _one_hot_encode_labels

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

from typing import Literal, Optional, Union

import torch
from torch import nn, optim
from torch.nn import functional as F

from .models import DenseNet121, ResNet50, MobileNetV3_Large

from torch.utils.data import DataLoader


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, eval_set: Literal["test", "val"], experiment):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.original_name = f"{self.model.__class__.__name__}_{eval_set}_{experiment}"
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(torch.argmax(label, dim = 1))
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=10_000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def _instantiate_model(model,
                       eval_set: Literal["test", "val"],
                       val_exp: str,
                       readout: Literal["RPE_Final", "RPE_classes", "Lens_Final", "Lens_classes"]):
    model_name = model.__class__.__name__
    model_file_name = f"{model_name}_{eval_set}_f1_{val_exp}_{readout}"
    print(f"...loading model {model_file_name}")
    state_dict_path = f"./classifiers/{model_file_name}_base_model.pth"
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    model.cuda()
    return model

def instantiate_model(model_name,
                      eval_set: Literal["test", "val"],
                      val_exp: str,
                      readout: Literal["RPE_Final", "RPE_classes", "Lens_Final", "Lens_classes"],
                      val_loader: DataLoader):
    num_classes = 4 if "classes" in readout else 2
    if model_name == "DenseNet121":
        model = DenseNet121(num_classes = num_classes, dropout = 0.2)
    elif model_name == "ResNet50":
        model = ResNet50(num_classes = num_classes, dropout = 0.2)
    elif model_name == "MobileNetV3_Large":
        model = MobileNetV3_Large(num_classes = num_classes, dropout = 0.5)
    else:
        raise ValueError("Unknown Model.")
    model = _instantiate_model(model = model,
                               eval_set = eval_set,
                               val_exp = val_exp,
                               readout = readout)
    model = ModelWithTemperature(model, eval_set, val_exp)
    model.set_temperature(val_loader)
    return model

def _read_dataset(dataset_id) -> Union[OrganoidDataset, OrganoidTrainingDataset]:
    return OrganoidDataset.read_classification_dataset(f"../raw_data/{dataset_id}")

def _create_dataloader(dataset, readout):
    return create_dataloader(dataset.X, dataset.y[readout], batch_size = 128, shuffle = False, train = False)

def _loop_to_timepoint(loops):
    return [int(loop.strip("LO")) for loop in loops]

def calculate_f1_scores(df):
    loops = df.sort_values(["loop"], ascending = True)["loop"].unique().tolist()
    f1_scores = pd.DataFrame(df.groupby('loop').apply(lambda x: f1_score(x['truth'], x['pred'], average = "weighted")), columns = ["F1"]).reset_index()
    f1_scores["loop"] = _loop_to_timepoint(loops)
    return f1_scores

def create_weights(val_scores, method='f1', normalize=True, power=1.0, temperature=1.0):
    """
    Creates weights for models based on validation F1 scores using various methodologies.

    Parameters:
    - val_scores (dict): Dictionary with model names as keys and F1 scores as values.
    - method (str): Method to use for weight calculation. Options are:
        - 'f1': Use F1 scores directly.
        - 'inverse': Use inverse of F1 scores.
        - 'softmax': Use softmax of F1 scores.
        - 'power': Use F1 scores raised to a power.
        - 'temperature': Use softmax with temperature scaling.
    - normalize (bool): Whether to normalize the weights to sum to 1.
    - power (float): Exponent to use in the 'power' method.
    - temperature (float): Temperature parameter for the 'temperature' method.

    Returns:
    - weights (dict): Dictionary with model names as keys and calculated weights as values.
    """

    # Extract model names and F1 scores
    model_names = list(val_scores.keys())
    f1_scores = np.array(list(val_scores.values()))
    
    if method == 'f1':
        # Use F1 scores directly
        weights_array = f1_scores.copy()
    elif method == 'softmax':
        # Use softmax of F1 scores
        exp_scores = np.exp(f1_scores)
        weights_array = exp_scores
    elif method == 'power':
        # Use F1 scores raised to a power
        weights_array = np.power(f1_scores, power)
    elif method == 'temperature':
        # Use softmax with temperature scaling
        exp_scores = np.exp(f1_scores / temperature)
        weights_array = exp_scores
    else:
        raise ValueError(f"Unknown method '{method}'. Available methods are 'f1', 'inverse', 'softmax', 'power', 'temperature'.")
    
    if normalize:
        total_weight = weights_array.sum()
        weights_array = weights_array / total_weight
    
    # Create weights dictionary
    weights = {model_name: weight for model_name, weight in zip(model_names, weights_array)}
    
    return weights

def ensemble_probability_averaging(models, dataloader, weights=None):
    """\
    Weights is a dictionary where the keys are the model names
    and the values is the val_f1_score.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_labels = []
    all_preds = []
    single_predictions = {model.original_name: [] for model in models}
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            prob_sum = 0
            total_weight = 0
            
            for model in models:
                logits = model(inputs)
                preds_single = torch.argmax(logits, dim=1).detach().cpu().numpy()
                single_predictions[model.original_name].append(preds_single)
                probs = F.softmax(logits, dim=1)
                
                if weights is not None:
                    # for now, lets allow keyerrors
                    weight = weights[model.original_name]
                    # weight = weights.get(model.original_name, 1.0)
                else:
                    # If no weights provided, use equal weighting
                    weight = 1.0
                
                # Accumulate the weighted probabilities
                prob_sum += weight * probs
                total_weight += weight
            
            # Normalize the accumulated probabilities
            probs_avg = prob_sum / total_weight
            # Determine the ensemble predictions
            preds = torch.argmax(probs_avg, dim=1)
            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_labels, all_preds, single_predictions


def generate_ensemble(val_experiments: list[str],
                      output_file: str,
                      model_names: list[str],
                      readout: Literal["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]):
    if os.path.isfile(output_file):
        with open(output_file, "rb") as file:
            models = pickle.load(file)
        return models

    else:

        models = []

        for experiment in val_experiments:
            print(f"\n{experiment}\n")

            # its necessary to load the val ID dataset for the TempScaling
            validation_dataset_id = f"{experiment}_full_SL3_fixed.cds"
            val_dataset = _read_dataset(validation_dataset_id)
            val_loader = _create_dataloader(val_dataset, readout)

            for model_name in model_names:
                print(f"\t{model_name}")
                # if eval_set == "both":
                models.append(
                    instantiate_model(
                        model_name,
                        eval_set = "test",
                        val_exp = experiment,
                        readout = readout,
                        val_loader = val_loader
                    )
                )
                if True:
                    pass
                else:
                    models.append(instantiate_model(model_name,
                                                    eval_set = "val",
                                                    val_exp = experiment,
                                                    readout = readout,
                                                    val_loader = val_loader))
        
        with open(output_file, "wb") as file:
            pickle.dump(models, file)

    return models


def neural_net_evaluation(cross_val_experiments: list[str],
                          val_experiment_id: str,
                          readout: Literal["RPE_Final", "RPE_classes", "Lens_Final", "Lens_classes"],
                          ensemble_output_file: str,
                          model_names: list[str],
                          eval_set: Literal["val", "test", "both"],
                          weights: Optional[dict] = None
                          ):
    labels = [0,1] if readout in ["RPE_Final", "Lens_Final"] else [0,1,2,3]
    if not isinstance(cross_val_experiments, list):
        cross_val_experiments = [cross_val_experiments]

    if eval_set == "test":
        validation_dataset_id = f"../raw_data/M{val_experiment_id}_full_SL3_fixed.cds"
        val_dataset = OrganoidDataset.read_classification_dataset(validation_dataset_id)
        val_dataset = OrganoidTrainingDataset(val_dataset, readout = readout)
        X_test, y_test = val_dataset.X_test, val_dataset.y_test
        val_loader = create_dataloader(X_test, y_test, batch_size = 128, shuffle = False, train = False)
    else:
        validation_dataset_id = f"{val_experiment_id}_full_SL3_fixed.cds"
        val_dataset = _read_dataset(validation_dataset_id)
        val_loader = _create_dataloader(val_dataset, readout)

    models = generate_ensemble(
        val_experiments = cross_val_experiments,
        readout = readout,
        output_file = ensemble_output_file,
        model_names = model_names,
    )
    truth_arr, ensemble_pred, single_predictions = ensemble_probability_averaging(models, val_loader, weights = weights)
    single_predictions = {
        key: np.hstack(single_predictions[key]) for key in single_predictions
    }
    assert -1 not in val_dataset.metadata["IMAGE_ARRAY_INDEX"]
    df = val_dataset.metadata

    if eval_set == "test":
        df = df[df["set"] == "test"]
    assert pd.Series(df["IMAGE_ARRAY_INDEX"]).is_monotonic_increasing

    print("metadata: ", pd.Series(df["loop"]).unique().shape)
    
    truth_values = pd.DataFrame(data = np.array([np.argmax(el) for el in truth_arr]),
                                columns = ["truth"],
                                index = df.index)
    df = pd.concat([df, truth_values], axis = 1)

    # ensemble F1
    pred_values = pd.DataFrame(data = ensemble_pred,
                                columns = ["pred"],
                                index = df.index)
    df = pd.concat([df, pred_values], axis = 1)

    f1_dfs = []
    
    conf_matrix_df = df.copy()
    print("conf matrix df: ", df["loop"].unique().shape)
    confusion_matrices = conf_matrix_df.groupby("loop").apply(
        lambda group: confusion_matrix(group["truth"], group["pred"], labels = labels)
    )
    print("pandas conf matrix series: ", confusion_matrices.shape)
    confusion_matrices = np.array(confusion_matrices.tolist())
    print("array: ", confusion_matrices.shape)

    conf_matrix = confusion_matrix(df["truth"].to_numpy(),
                                   df["pred"].to_numpy(),
                                   labels = labels)
    ensemble_f1 = calculate_f1_scores(df)
    ensemble_f1 = ensemble_f1.rename(columns = {"F1": "Ensemble"}).set_index("loop")
    f1_dfs.append(ensemble_f1)

    for model in single_predictions:
        df.loc[:, "pred"] = single_predictions[model]
        f1 = calculate_f1_scores(df)
        f1 = f1.rename(columns = {"F1": model}).set_index("loop")
        f1_dfs.append(f1)

    neural_net_f1 = pd.concat(f1_dfs, axis = 1)
    neural_net_f1 = neural_net_f1.reset_index().melt(id_vars = "loop",
                                                     value_name = "F1",
                                                     var_name = "Neural Net")

    return neural_net_f1, conf_matrix, confusion_matrices

def _get_unique_experiment_well_combo(df: pd.DataFrame,
                                      col1: str,
                                      col2: str):
    return df[[col1, col2]].drop_duplicates().reset_index(drop=True).to_numpy()

def _filter_wells(df: pd.DataFrame,
                  combinations: np.ndarray,
                  columns: list[str]):
    combinations_df = pd.DataFrame(combinations, columns=columns)
    return df.merge(combinations_df, on=columns, how='inner')

def _calculate_train_test_split(df):
    unique_well_per_experiment = _get_unique_experiment_well_combo(df, "experiment", "well")
    train_wells, test_wells = train_test_split(
        unique_well_per_experiment,
        test_size = 0.1,
        random_state = 187
    )
    assert isinstance(train_wells, np.ndarray)
    assert isinstance(test_wells, np.ndarray)
    train_df = _filter_wells(df, train_wells, ["experiment", "well"])
    test_df = _filter_wells(df, test_wells, ["experiment", "well"])
    return train_df, test_df

def _assemble_morphometrics_dataframe(train_experiments: list[str],
                                      val_experiment_id: str,
                                      readout: str,
                                      eval_set: Literal["test", "val"]):
    metadata_columns = [
        'experiment', 'well', 'file_name', 'position', 'slice', 'loop',
        'Condition', 'RPE_Final', 'RPE_Norin', 'RPE_Cassian',
        'Confidence_score_RPE', 'Total_RPE_amount', 'RPE_classes', 'Lens_Final',
        'Lens_Norin', 'Lens_Cassian', 'Confidence_score_lens', 'Lens_area',
        'Lens_classes', 'label'
    ]
    
    frame = None
    for i, exp in enumerate(train_experiments):
        data = pd.read_csv(f"../../shape_analysis/results/{exp}_morphometrics.csv")
        if i == 0:
            frame = data
        else:
            frame = pd.concat([frame, data], axis = 0)
    assert frame is not None
    
    # we remove columns and rows that only contain NA
    frame = frame.dropna(how = "all", axis = 1)

    data_columns = [col for col in frame.columns if col not in metadata_columns]  
    frame = frame.dropna(how = "all", axis = 0, subset = data_columns)
    frame = frame[~frame["RPE_classes"].isna()]

    # we dont need label
    frame = frame.drop("label", axis = 1)

    assert not frame.isna().any().sum()

    data_columns = [col for col in frame.columns if col not in metadata_columns]

    df = frame

    non_val_df = df[df["experiment"] != val_experiment_id].copy()
    val_df = df[df["experiment"] == val_experiment_id].copy()

    assert isinstance(non_val_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    scaler = StandardScaler()
    scaler.fit(non_val_df[data_columns])

    non_val_df[data_columns] = scaler.transform(non_val_df[data_columns])
    val_df[data_columns] = scaler.transform(val_df[data_columns])

    second_scaler = MinMaxScaler()
    second_scaler.fit(non_val_df[data_columns])

    non_val_df[data_columns] = second_scaler.transform(non_val_df[data_columns])
    val_df[data_columns] = second_scaler.transform(val_df[data_columns])

    if eval_set == "val":
        X_train = non_val_df.copy()
        y_train = _one_hot_encode_labels(X_train[readout].to_numpy(),
                                         readout = readout)
        train_truth = pd.DataFrame(data = np.argmax(y_train, axis = 1),
                                   columns = ["truth"],
                                   index = X_train.index)
        X_train = pd.concat([X_train, train_truth], axis = 1)

        X_test = pd.DataFrame()
        y_test = np.array([])
        
        X_val = val_df.copy()
        y_val = _one_hot_encode_labels(X_val[readout].to_numpy(),
                                       readout = readout)
        val_truth = pd.DataFrame(data = np.argmax(y_val, axis = 1),
                                    columns = ["truth"],
                                    index = val_df.index)
        X_val = pd.concat([X_val, val_truth], axis = 1)
    else:
        train_df, test_df = _calculate_train_test_split(non_val_df)

        X_train = train_df
        y_train = _one_hot_encode_labels(X_train[readout].to_numpy(),
                                         readout = readout)
        train_truth = pd.DataFrame(data = np.argmax(y_train, axis = 1),
                                   columns = ["truth"],
                                   index = X_train.index)
        X_train = pd.concat([X_train, train_truth], axis = 1)

        X_test = test_df
        y_test = _one_hot_encode_labels(X_test[readout].to_numpy(),
                                         readout = readout)
        test_truth = pd.DataFrame(data = np.argmax(y_test, axis = 1),
                                  columns = ["truth"],
                                  index = X_test.index)
        X_test = pd.concat([X_test, test_truth], axis = 1)

        X_val = val_df.copy()
        y_val = _one_hot_encode_labels(X_val[readout].to_numpy(),
                                       readout = readout)
        val_truth = pd.DataFrame(data = np.argmax(y_val, axis = 1),
                                 columns = ["truth"],
                                 index = X_val.index)
        val_df = pd.concat([val_df, val_truth], axis = 1)

    return X_train, y_train, X_test, y_test, X_val, y_val, data_columns

def _get_best_params(classifier: str,
                     readout: str) -> dict:
    with open(f"../../shape_analysis/results/best_params/best_params_{classifier}_{readout}.dict", "rb") as file:
        best_params_ = pickle.load(file)

    return best_params_


def _get_classifier(readout):
    if readout == "RPE_Final":
        best_params = _get_best_params("SGDClassifier", readout)
        return MultiOutputClassifier(SGDClassifier(**best_params), n_jobs = 16)
    elif readout == "Lens_Final":
        best_params = _get_best_params("SGDClassifier", readout)
        return MultiOutputClassifier(SGDClassifier(**best_params), n_jobs = 16)
    elif readout == "RPE_classes":
        best_params = _get_best_params("DecisionTreeClassifier", readout)
        return DecisionTreeClassifier(**best_params)
    elif readout == "Lens_classes":
        best_params = _get_best_params("KNN", readout)
        if "n_jobs" not in best_params:
            best_params["n_jobs"] = 16
        return KNeighborsClassifier(**best_params)
    else:
        raise ValueError("Unknown readout")

def classifier_evaluation(train_experiments,
                          val_experiment_id,
                          readout,
                          classifier: str,
                          eval_set: Literal["test", "val"]):
    labels = [0,1] if readout in ["RPE_Final", "Lens_Final"] else [0,1,2,3]
    X_train, y_train, X_test, y_test, X_val, y_val, data_columns = _assemble_morphometrics_dataframe(
        train_experiments,
        val_experiment_id,
        readout,
        eval_set
    )
    clf = _get_classifier(readout)

    clf.fit(X_train[data_columns], y_train)
    if eval_set == "val":
        pred_values = pd.DataFrame(data = np.argmax(clf.predict(X_val[data_columns]), axis = 1),
                                   columns = ["pred"],
                                   index = X_val.index)
        result = pd.concat([X_val, pred_values], axis = 1)
    else:
        pred_values = pd.DataFrame(data = np.argmax(clf.predict(X_test[data_columns]), axis = 1),
                                   columns = ["pred"],
                                   index = X_test.index)
        result = pd.concat([X_test, pred_values], axis = 1)

    result = result.sort_values(["experiment", "well", "loop", "slice"], ascending = [True, True, True, True])

    f1_scores = calculate_f1_scores(result)
    conf_matrix = confusion_matrix(result["truth"].to_numpy(), result["pred"].to_numpy(), labels = labels)
    confusion_matrices = result.groupby("loop").apply(
        lambda group: confusion_matrix(group["truth"], group["pred"], labels = labels)
    )
    confusion_matrices = np.array(confusion_matrices.tolist())


    return f1_scores, conf_matrix, confusion_matrices

# def accumulative_prediction(pred_values):
#     pred_values = pred_values.to_numpy()
#     pred_values = np.hstack([pred_values, np.array([0,1])])
#     pred_one_hot = OneHotEncoder().fit_transform(pred_values.reshape(pred_values.shape[0], 1)).toarray()
#     pred_one_hot = pred_one_hot[:-2]
#     pred_values = pred_values[:-2]
#     assert pred_values.shape[0] == pred_one_hot.shape[0], (pred_values.shape, pred_one_hot.shape)
#     acc_prediction = []
#     for i in range(1, pred_values.shape[0]+1):
#         subarr = pred_one_hot[i-5:i]
#         acc_prediction.append(np.argmax(np.sum(subarr, axis = 0)))
#     acc_pred_arr = np.array(acc_prediction).flatten()
#     return acc_pred_arr


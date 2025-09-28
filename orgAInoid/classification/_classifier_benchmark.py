import os
import pandas as pd

from sklearn.multioutput import MultiOutputClassifier

from ._classifier_scoring import SCORES_TO_USE, write_to_scores
from .models import CLASSIFIERS_TO_TEST_FULL

from ._utils import _run_classifier_fit_and_val

from typing import Optional


def run_classifier_comparison(
    df: pd.DataFrame,
    output_dir: str,
    data_columns: list[str],
    readouts: Optional[list[str]] = None,
    analysis_id: Optional[str] = None,
):
    """\
    Function to run the classifier benchmark.

    df is the dataframe with data_columns consisting of unscaled, raw data.

    The function will run through the classifiers defined in .models.

    Classifiers that have been calculated already will be skipped.

    """

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = (
        "algorithm,readout,score_on,experiment,train_time,"
        + f"pred_time_train,pred_time_test,pred_time_val,{scores}\n"
    )
    if analysis_id is not None:
        score_key = f"CLFCOMP_{analysis_id}"
    else:
        score_key = "Scores"
    score_file = os.path.join(output_dir, f"{score_key}.log")
    if not os.path.isfile(score_file):
        write_to_scores(
            resource_metrics, output_dir=output_dir, key=score_key, init=True
        )
        already_calculated = {}
    else:
        scores = pd.read_csv(score_file, index_col=False)
        scores = scores[["algorithm", "readout"]].drop_duplicates()
        already_calculated = {}
        for _readout in scores["readout"].unique():
            already_calculated[_readout] = scores.loc[
                scores["readout"] == _readout, "algorithm"
            ].tolist()
    if readouts is None:
        readouts = ["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]
    elif not isinstance(readouts, list):
        readouts = [readouts]

    experiments = df["experiment"].unique().tolist()

    for readout in readouts:
        for classifier in CLASSIFIERS_TO_TEST_FULL:
            if readout in already_calculated:
                if classifier in already_calculated[readout]:
                    print(
                        f"Skipping {classifier} for {readout} as it is already calculated"
                    )
                    continue

            if classifier in ["LabelPropagation", "LabelSpreading", "CategoricalNB"]:
                print(f"Skipping {classifier} due to memory reasons!")
                continue

            print(f"... running {classifier} on readout {readout}")
            for experiment in experiments:
                if CLASSIFIERS_TO_TEST_FULL[classifier]["allows_multi_class"]:
                    if CLASSIFIERS_TO_TEST_FULL[classifier]["multiprocessing"]:
                        clf = CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"](
                            n_jobs=16
                        )
                    else:
                        clf = CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"]()
                else:
                    if CLASSIFIERS_TO_TEST_FULL[classifier]["scalable"] is False:
                        clf = MultiOutputClassifier(
                            CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"]()
                        )
                    else:
                        clf = MultiOutputClassifier(
                            CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"](),
                            n_jobs=16,
                        )

                _run_classifier_fit_and_val(
                    df=df,
                    experiment=experiment,
                    data_columns=data_columns,
                    readout=readout,
                    clf=clf,
                    classifier_name=classifier,
                    output_dir=output_dir,
                    score_key=score_key,
                )

    return

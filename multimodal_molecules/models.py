import json
from pathlib import Path
import pickle
from time import perf_counter
from warnings import warn

from monty.json import MSONable
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from multimodal_molecules.data import (
    get_dataset,
    get_reproducible_train_test_split,
)


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def dt(self):
        return self._time


def rf_classifier_predict(rf, x):
    return np.array([tree.predict(x) for tree in rf.estimators_])


class Report(MSONable):
    @property
    def metrics(self):
        return self._metrics

    def __init__(self, metrics=dict()):
        self._metrics = metrics

    def populate_performance(self, model, x_train, y_train, x_test, y_test):
        y_test_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)

        self._metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)
        self._metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)

        self._metrics["test_balanced_accuracy"] = balanced_accuracy_score(
            y_test, y_test_pred
        )
        self._metrics["train_balanced_accuracy"] = balanced_accuracy_score(
            y_train, y_train_pred
        )

    def populate_feature_importance(self, model, x_test, y_test, n_jobs):
        self._metrics["feature_importance"] = np.array(
            [tree.feature_importances_ for tree in model.estimators_]
        )
        p_importance = permutation_importance(
            model, x_test, y_test, n_jobs=n_jobs
        )
        p_importance.pop("importances")
        self._metrics["permutation_feature_importance"] = p_importance


def run_experiments(
    xanes_path,
    index_path,
    conditions,
    root=".",
    offset_left=None,
    offset_right=None,
    test_size=0.9,
    random_state=42,
    n_jobs=2,
    testing=False,
    min_fg_occurrence=0.02,
    max_fg_occurrence=0.98,
):
    """Summary

    Parameters
    ----------
    xanes_path : TYPE
        Description
    index_path : TYPE
        Description
    conditions : TYPE
        Description
    root : str, optional
        Description
    offset_left : None, optional
        Description
    offset_right : None, optional
        Description
    test_size : float, optional
        Description
    random_state : int, optional
        Description
    n_jobs : int, optional
        Description
    testing : bool, optional
        Description
    min_fg_occurrence : float, optional
        Description
    max_fg_occurrence : float, optional
        Description
    """

    Path(root).mkdir(exist_ok=True, parents=True)

    # "Sort" the conditions
    conditions = ",".join(sorted(conditions.split(",")))

    # For a given set of conditions, retrieve the relevant data
    data = get_dataset(xanes_path, index_path, conditions)

    # Select the keys that contain the substring "XANES"
    xanes_keys_avail = [cc for cc in conditions.split(",") if "XANES" in cc]

    xanes_data = np.concatenate(
        [data[key][:, offset_left:offset_right] for key in xanes_keys_avail],
        axis=1,
    )

    base_name = conditions.replace(",", "_")

    print(f"Total XANES data has shape {xanes_data.shape}")
    L = len(data["FG"])
    print(f"Total of {L} functional groups")
    for ii, (key, binary_targets) in enumerate(data["FG"].items()):

        # Check that the occurence of the functional groups falls into the
        # specified sweet spot
        total_targets = len(binary_targets)
        targets_on = binary_targets.sum()
        ratio = targets_on / total_targets
        if not (min_fg_occurrence < ratio < max_fg_occurrence):
            print(f"{key} has occurence of {ratio:.02f} - continuing")
            continue

        # Get the name of the model to be trained. This should correspond to
        # a set of conditions (XANES, kept data, etc.) and a functional group
        # to be predicted
        name = f"{base_name}_{key}"

        print(f"Running {name}...")

        with Timer() as timer:

            # Get the test/train split
            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = get_reproducible_train_test_split(
                xanes_data,
                binary_targets,
                test_size=test_size,
                random_state=random_state,
            )

            # Train the model
            model = RandomForestClassifier(
                n_jobs=n_jobs, random_state=random_state
            )
            model.fit(x_train, y_train)

        print(f"- ({ii+1}/{L}) training: {int(timer.dt)} s")

        with Timer() as timer:
            # Run the model report and save the it as a json file
            report = Report()
            report.populate_performance(
                model, x_train, y_train, x_test, y_test
            )
            report.populate_feature_importance(model, x_test, y_test, n_jobs)

            report_path = Path(root) / f"{name}.json"
            with open(report_path, "w") as f:
                json.dump(report.to_json(), f, indent=4)

            # Save the model itself
            model_path = Path(root) / f"{name}_model.pkl"
            pickle.dump(
                model,
                open(model_path, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"- ({ii+1}/{L}) report/save: {int(timer.dt)} s")

        if testing:
            warn("In testing mode, only computing one model!")
            break


def load_results(
    xanes_path,
    index_path,
    conditions,
    root=".",
    populate_models=False,
):
    """Summary

    Parameters
    ----------
    xanes_path : TYPE
        Description
    index_path : TYPE
        Description
    conditions : TYPE
        Description
    root : str, optional
        Description
    populate_models : bool, optional
        Description

    Returns
    -------
    dict
    """

    # "Sort" the conditions
    conditions = ",".join(sorted(conditions.split(",")))

    # For a given set of conditions, retrieve the relevant data
    data = get_dataset(xanes_path, index_path, conditions)

    # Load in all of the data that matches this conditions pattern
    base_name = conditions.replace(",", "_")
    reports_list = Path(root).glob(f"{base_name}*.json")
    results = dict()
    models = dict()
    for path in reports_list:
        with open(path, "r") as f:
            d = json.loads(json.load(f))
        results[path.stem] = Report.from_dict(d)
        if populate_models:
            parent = path.parent
            pkl_file = Path(parent) / f"{base_name}_model.pkl"
            models[path.stem] = pickle.load(open(pkl_file))

    return dict(data=data, results=results, models=models)

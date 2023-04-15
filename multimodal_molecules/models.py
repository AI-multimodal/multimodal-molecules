from itertools import combinations
from functools import cached_property, cache
import json
from pathlib import Path
import pickle
import random
from time import perf_counter
from tqdm import tqdm
from warnings import warn

from monty.json import MSONable
import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multimodal_molecules.data import get_dataset


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def dt(self):
        return self._time


def get_all_combinations(n):
    L = [ii for ii in range(n)]
    combos = []
    for nn in range(len(L)):
        combos.extend(list(combinations(L, nn + 1)))
    return combos


def predict_rf(rf, X):
    return np.array([est.predict(X) for est in rf.estimators_]).T


class Results(MSONable):
    """A full report for all functional groups given a set of conditions."""

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f:
            d = json.loads(json.load(f))
        klass = cls.from_dict(d)
        klass._data_loaded_from = str(Path(path).parent)
        return klass

    @property
    def report(self):
        return self._report

    @cache
    def get_model(self, key):

        if self._data_loaded_from is None:
            warn(
                "Use this after loading from json and saving the pickled "
                "models. Returning None"
            )
            return None

        model_path = Path(self._data_loaded_from) / f"{key}_model.pkl"
        return pickle.load(open(model_path, "rb"))

    @cached_property
    def train_val_test_indexes(self):
        if self._data_size is None:
            raise RuntimeError("Run experiments first to calculate data size")

        random.seed(self._random_state)
        N = self._data_size
        indexes = [ii for ii in range(N)]
        random.shuffle(indexes)
        test_size = int(self._test_size * N)
        val_size = int(self._val_size * N)
        t_plus_v = test_size + val_size

        test_indexes = indexes[:test_size]
        val_indexes = indexes[test_size:t_plus_v]
        train_indexes = indexes[t_plus_v:]

        assert set(train_indexes).isdisjoint(set(val_indexes))
        assert set(train_indexes).isdisjoint(set(test_indexes))
        assert set(test_indexes).isdisjoint(set(val_indexes))

        return sorted(train_indexes), sorted(val_indexes), sorted(test_indexes)

    @cache
    def get_data(self, input_data_directory):
        xanes_path = Path(input_data_directory) / self._xanes_data_name
        index_path = Path(input_data_directory) / self._index_data_name
        print(f"Loading data from {input_data_directory}")
        return get_dataset(xanes_path, index_path, self._conditions)

    def _get_xanes_data(self, data):
        """Select the keys that contain the substring "XANES". Also returns
        the lenght of the keys available."""

        xanes_keys_avail = [
            cc for cc in self._conditions.split(",") if "XANES" in cc
        ]
        o1 = self._offset_left
        o2 = self._offset_right
        return np.concatenate(
            [data[key][:, o1:o2] for key in xanes_keys_avail],
            axis=1,
        ), len(xanes_keys_avail)

    def get_train_test_split(self, data, xanes="C,N,O"):
        """Gets the training and testing splits from provided data.

        Parameters
        ----------
        data : dict
            The data as loaded by the ``get_dataset`` function.
        xanes : str, optional
            Description

        Returns
        -------
        dict
            A dictionary containing the trainind/testing data for this set of
            results.
        """

        xanes = [f"{xx}-XANES" for xx in xanes.split(",")]
        conditions = self._conditions.split(",")
        assert set(xanes).issubset(set(conditions))
        indexes = [conditions.index(xx) for xx in xanes]

        train_idx, val_idx, test_idx = self.train_val_test_indexes

        xanes_data, n_xanes_types = self._get_xanes_data(data)
        ssl = xanes_data.shape[1] // n_xanes_types  # Single spectrum length

        current_xanes_data = np.concatenate(
            [
                xanes_data[:, ssl * ii : ssl * (ii + 1)] for ii in indexes
            ],  # noqa
            axis=1,
        )

        xanes_data_train = current_xanes_data[train_idx, :]
        xanes_data_val = current_xanes_data[val_idx, :]
        xanes_data_test = current_xanes_data[test_idx, :]

        functional_groups = data["FG"]
        keys = functional_groups.keys()
        train_fg = {key: functional_groups[key][train_idx] for key in keys}
        val_fg = {key: functional_groups[key][val_idx] for key in keys}
        test_fg = {key: functional_groups[key][test_idx] for key in keys}

        return {
            "x_train": xanes_data_train,
            "x_val": xanes_data_val,
            "x_test": xanes_data_test,
            "y_train": train_fg,
            "y_val": val_fg,
            "y_test": test_fg,
            "unique_functional_groups": list(functional_groups),
        }

    def __init__(
        self,
        conditions,
        xanes_data_name="221205_xanes.pkl",
        index_data_name="221205_index.csv",
        offset_left=None,
        offset_right=None,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        min_fg_occurrence=0.02,
        max_fg_occurrence=0.98,
        data_size=None,
        data_loaded_from=None,
        report=None,
        specific_functional_groups=None,
        pca_components=0,
        nmf_components=0,
        rf_kwargs={},
    ):
        self._conditions = ",".join(sorted(conditions.split(",")))
        self._xanes_data_name = xanes_data_name
        self._index_data_name = index_data_name
        self._offset_left = offset_left
        self._offset_right = offset_right
        self._val_size = val_size
        self._test_size = test_size
        self._random_state = random_state
        self._min_fg_occurrence = min_fg_occurrence
        self._max_fg_occurrence = max_fg_occurrence
        self._data_size = data_size
        self._data_loaded_from = data_loaded_from
        self._rf_kwargs = rf_kwargs
        self._specific_functional_groups = specific_functional_groups

        if pca_components > 0 and nmf_components > 0:
            raise ValueError("Choose one PCA or NMF, not both")

        self._pca_components = pca_components
        self._nmf_components = nmf_components

        # There was a reason for doing this but I can't remember what it was
        # I.e. I don't recommend just setting report={} as a kwarg default...
        if report is None:
            self._report = {}
        else:
            self._report = report

    def run_experiments(
        self,
        input_data_directory="data/221205",
        output_data_directory=None,
        n_jobs=2,
        debug=-1,
        compute_feature_importance=False,
    ):
        """Runs all experiments corresponding to the functional groups
        and the initially provided conditions.

        Parameters
        ----------
        input_data_directory : str
            The location of the input data. Should contain the xanes.pkl-like
            file and the index.csv-like file. The specific names of these files
            are provided at class instantiation.
        output_data_directory : os.PathLike, optional
            The location of the target directory for saving results. If None,
            no results are saved to disk and must be done manually.
        n_jobs : int, optional
            The number of jobs/parallel processes to feed to the RandomForest
            model and the feature impotance ranking functions.
        debug : bool, optional
            If >0, iterates only through that many calculations.
        compute_feature_importance : bool, optional
            Computes the feature importances using the permutation method.
            Note that this is quite expensive. Likely to take aroudn 2 minutes
            or so per model even at full parallelization. This is probably best
            set to False since run_experiments evaluates on the validation
            data, and the permutation feature importance only needs to be
            evaluated on the testing data.
        """

        SPACING = "\t          "

        print("--------------------------------------------------------------")

        data = self.get_data(input_data_directory)
        xanes_data, n_xanes_types = self._get_xanes_data(data)
        self._data_size = xanes_data.shape[0]
        ssl = xanes_data.shape[1] // n_xanes_types  # Single spectrum length
        train_indexes, val_indexes, test_indexes = self.train_val_test_indexes

        base_name = self._conditions.replace(",", "_")

        if self._specific_functional_groups is None:
            functional_groups = data["FG"]
        else:
            try:
                functional_groups = {
                    key: data["FG"][key]
                    for key in self._specific_functional_groups
                }
            except KeyError as err:
                print("Available functional groups:", list(data["FG"].keys()))
                raise KeyError(err)

        print(f"Total XANES data has shape {xanes_data.shape}")
        L = len(functional_groups)
        print(f"Total of {L} functional groups")

        xanes_index_combinations = get_all_combinations(n_xanes_types)

        root = Path(output_data_directory)
        conditions_list = base_name.split("_")
        for jj, combo in enumerate(xanes_index_combinations):
            current_conditions_name = "_".join(
                [conditions_list[jj] for jj in combo]
            )
            current_xanes_data = np.concatenate(
                [
                    xanes_data[:, ssl * ii : ssl * (ii + 1)] for ii in combo
                ],  # noqa
                axis=1,
            )
            print(
                f"\nCurrent XANES combo={combo} name={current_conditions_name} "
                f"shape={current_xanes_data.shape}"
            )

            for ii, (fg_name, binary_targets) in enumerate(
                functional_groups.items()
            ):

                ename = f"{current_conditions_name}-{fg_name}"
                print(f"\n\t[{(ii+1):03}/{L:03}] {ename}")

                model_path = root / f"{ename}_model.pkl"
                if Path(model_path).exists():
                    print(f"\tmodel={model_path} exists, continuing")
                    continue

                # Check that the occurence of the functional groups falls into
                # the specified sweet spot
                p_total = binary_targets.sum() / len(binary_targets)
                if not (
                    self._min_fg_occurrence < p_total < self._max_fg_occurrence
                ):
                    print(
                        f"{SPACING}Occ. {p_total:.04f} - continuing\n",
                        flush=True,
                    )
                    continue

                x_train = current_xanes_data[train_indexes, :]
                x_val = current_xanes_data[val_indexes, :]
                y_train = binary_targets[train_indexes]
                y_val = binary_targets[val_indexes]

                p_val = y_val.sum() / len(y_val)
                p_train = y_train.sum() / len(y_train)

                if jj == 0:
                    print(
                        f"{SPACING}"
                        f"Occ. total={p_total:.04f} | train={p_train:.04f} | "
                        f"val={p_val:.04f} "
                    )

                # Train the model
                with Timer() as timer:

                    model = RandomForestClassifier(
                        n_jobs=n_jobs,
                        random_state=self._random_state,
                        **self._rf_kwargs,
                    )

                    if self._pca_components > 0:
                        model = Pipeline(
                            steps=[
                                ("scaler", StandardScaler()),
                                (
                                    "pca",
                                    PCA(self._pca_components * len(combo)),
                                ),
                                ("rf", model),
                            ]
                        )

                    elif self._nmf_components > 0:
                        model = Pipeline(
                            steps=[
                                (
                                    "nmf",
                                    NMF(self._nmf_components * len(combo)),
                                ),
                                ("rf", model),
                            ]
                        )

                    model.fit(x_train, y_train)

                print(f"{SPACING}Training: {timer.dt:.01f} s")

                with Timer() as timer:
                    y_val_pred = model.predict(x_val)
                    y_train_pred = model.predict(x_train)

                    # Accuracies and other stuff
                    self._report[ename] = {
                        "p_total": p_total,
                        "p_train": p_train,
                        "p_val": p_val,
                        "val_accuracy": accuracy_score(y_val, y_val_pred),
                        "train_accuracy": accuracy_score(
                            y_train, y_train_pred
                        ),
                        "val_balanced_accuracy": balanced_accuracy_score(
                            y_val, y_val_pred
                        ),
                        "train_balanced_accuracy": balanced_accuracy_score(
                            y_train, y_train_pred
                        ),
                    }

                    if compute_feature_importance:
                        # Standard feature importance from the RF model
                        # This is very fast
                        f_importance = np.array(
                            [
                                tree.feature_importances_
                                for tree in model.estimators_
                            ]
                        )

                        # Append to the report
                        self._report[ename]["feature_importance"] = {
                            "importances_mean": f_importance.mean(axis=0),
                            "importances_std": f_importance.std(axis=0),
                        }

                        # More accurate permutation feature importance
                        p_importance = permutation_importance(
                            model, x_val, y_val, n_jobs=n_jobs
                        )
                        p_importance.pop("importances")

                        # Append to the report
                        self._report[ename][
                            "permutation_feature_importance"
                        ] = {
                            "importances_mean": p_importance.mean(axis=0),
                            "importances_std": p_importance.std(axis=0),
                        }

                print(f"{SPACING}Report/save: {timer.dt:.01f} s")
                val_acc = self._report[ename]["val_balanced_accuracy"]
                train_acc = self._report[ename]["train_balanced_accuracy"]
                print(
                    f"{SPACING}Class-balanced accuracies: "
                    f"train={train_acc:.05f} | "
                    f"val={val_acc:.05f}",
                    flush=True,
                )

                if output_data_directory is not None:

                    # Save the model itself
                    pickle.dump(
                        model,
                        open(model_path, "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    print(f"{SPACING}Model saved to {model_path}")

                if debug > 0:
                    if ii >= debug:
                        print("In testing mode- ending early!", flush=True)
                        break

        if output_data_directory is not None:
            Path(output_data_directory).mkdir(exist_ok=True, parents=True)
            report_path = root / f"{base_name}.json"
            with open(report_path, "w") as f:
                json.dump(self.to_json(), f, indent=4, sort_keys=True)
            print(f"\nReport saved to {report_path}")


def validate(path, data_directory):
    """A helper function for validating that the results of the models
    (pickled) are the same as those which were stored in the reports.

    Parameters
    ----------
    path : os.PathLike
        Path to the json file which contains the model results.
    input_data_directory : TYPE
        Description
    """

    results = Results.from_file(path)
    xanes_data_path = Path(data_directory) / results._xanes_data_name
    index_data_path = Path(data_directory) / results._index_data_name
    conditions = results._conditions
    data = get_dataset(xanes_data_path, index_data_path, conditions)
    _, val_idx, _ = results.train_val_test_indexes

    for key in tqdm(results.report.keys()):

        model = results.get_model(key)

        # Get the XANES keys
        xk = [xx for xx in key.split("XANES")[:-1]]
        xk = [xx.replace("-", "").replace("_", "") for xx in xk]
        xk = [f"{xx}-XANES" for xx in xk]

        # Get the input data
        o1 = results._offset_left
        o2 = results._offset_right
        X = np.concatenate(
            [data[key][:, o1:o2] for key in xk],
            axis=1,
        )

        # Get the predictions and the ground truth for the model
        preds = model.predict(X[val_idx, :])
        fg = key.split("XANES-")[-1]
        targets = data["FG"][fg][val_idx]
        balanced_acc = balanced_accuracy_score(targets, preds)

        # Get the previously cached results for the accuracy
        previous_balanced_acc = results.report[key]["val_balanced_accuracy"]

        # print(f"{previous_balanced_acc:.02f} | {balanced_acc:.02f}")
        assert np.allclose(balanced_acc, previous_balanced_acc)

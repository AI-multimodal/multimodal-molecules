from copy import deepcopy
from functools import cached_property, cache
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

        # Accuracies
        self._metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)
        self._metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)

        # Balanced accuracies
        self._metrics["test_balanced_accuracy"] = balanced_accuracy_score(
            y_test, y_test_pred
        )
        self._metrics["train_balanced_accuracy"] = balanced_accuracy_score(
            y_train, y_train_pred
        )

    def populate_feature_importance(self, model, x_test, y_test, n_jobs):

        # Standard feature importance from the RF model
        f_importance = np.array(
            [tree.feature_importances_ for tree in model.estimators_]
        )
        self._metrics["feature_importance"] = {
            "importances_mean": f_importance.mean(axis=0),
            "importances_std": f_importance.std(axis=0),
        }

        # More accurate permutation feature importance
        p_importance = permutation_importance(
            model, x_test, y_test, n_jobs=n_jobs
        )
        p_importance.pop("importances")
        self._metrics["permutation_feature_importance"] = p_importance


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
    def reports(self):
        return self._reports

    @cached_property
    def models(self):
        """Returns a dictionary of the loaded models. Note this requires that
        _data_loaded_from is set. This only happens when loading the class from
        a json file.

        Returns
        -------
        dict
        """

        if self._data_loaded_from is None:
            warn(
                "Use this after loading from json and saving the pickled "
                "models. Returning None"
            )
            return None

        base_name = self._conditions.replace(",", "_")
        path = Path(self._data_loaded_from) / f"{base_name}_models.pkl"
        return pickle.load(open(path, "rb"))

        d = dict()
        for fname in Path(self._data_loaded_from).glob("*.pkl"):
            key = str(fname).split(".pkl")[0].split("_")[1]
            d[key] = pickle.load(open(fname, "rb"))
        return d

    @cached_property
    def train_test_indexes(self):
        if self._data_size is None:
            raise RuntimeError("Run experiments first to calculate data size")
        np.random.seed(self._random_state)
        N = self._data_size
        size = int(self._test_size * N)
        test_indexes = np.random.choice(N, size=size, replace=False).tolist()
        assert len(test_indexes) == len(np.unique(test_indexes))
        train_indexes = list(set(np.arange(N).tolist()) - set(test_indexes))
        assert set(test_indexes).isdisjoint(set(train_indexes))
        return sorted(train_indexes), sorted(test_indexes)

    @cache
    def get_data(self, input_data_directory):
        xanes_path = Path(input_data_directory) / self._xanes_data_name
        index_path = Path(input_data_directory) / self._index_data_name
        return get_dataset(xanes_path, index_path, self._conditions)

    def _get_xanes_data(self, data):
        """Select the keys that contain the substring "XANES"."""

        xanes_keys_avail = [
            cc for cc in self._conditions.split(",") if "XANES" in cc
        ]
        o1 = self._offset_left
        o2 = self._offset_right
        return np.concatenate(
            [data[key][:, o1:o2] for key in xanes_keys_avail],
            axis=1,
        )

    def __init__(
        self,
        conditions,
        xanes_data_name="221205_xanes.pkl",
        index_data_name="221205_index.csv",
        offset_left=None,
        offset_right=None,
        test_size=0.9,
        random_state=42,
        min_fg_occurrence=0.02,
        max_fg_occurrence=0.98,
        data_size=None,
        reports=dict(),
        data_loaded_from=None,
    ):
        self._conditions = ",".join(sorted(conditions.split(",")))
        self._xanes_data_name = xanes_data_name
        self._index_data_name = index_data_name
        self._offset_left = offset_left
        self._offset_right = offset_right
        self._test_size = test_size
        self._random_state = random_state
        self._min_fg_occurrence = min_fg_occurrence
        self._max_fg_occurrence = max_fg_occurrence
        self._data_size = data_size
        self._reports = reports
        self._data_loaded_from = data_loaded_from

    def run_experiments(
        self,
        input_data_directory="data/221205",
        output_data_directory=None,
        n_jobs=2,
        debug=-1,
        compute_feature_importance=True,
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
            or so per model even at full parallelization.
        """

        data = self.get_data(input_data_directory)
        xanes_data = self._get_xanes_data(data)
        self._data_size = xanes_data.shape[0]

        base_name = self._conditions.replace(",", "_")

        print(f"Total XANES data has shape {xanes_data.shape}")
        L = len(data["FG"])
        print(f"Total of {L} functional groups")

        train_indexes, test_indexes = self.train_test_indexes

        models = dict()

        for ii, (key, binary_targets) in enumerate(data["FG"].items()):

            # Check that the occurence of the functional groups falls into the
            # specified sweet spot
            total_targets = len(binary_targets)
            targets_on = binary_targets.sum()
            ratio = targets_on / total_targets
            if not (self._min_fg_occurrence < ratio < self._max_fg_occurrence):
                print(
                    f"[{(ii+1):03}/{L:03}] {key} occurence {ratio:.04f} "
                    "- continuing"
                )
                continue

            print(
                f"[{(ii+1):03}/{L:03}] {key} occurence {ratio:.04f} ", end=""
            )

            with Timer() as timer:

                x_train = xanes_data[train_indexes, :]
                x_test = xanes_data[test_indexes, :]
                y_train = binary_targets[train_indexes]
                y_test = binary_targets[test_indexes]

                # Train the model
                model = RandomForestClassifier(
                    n_jobs=n_jobs, random_state=self._random_state
                )
                model.fit(x_train, y_train)

                if output_data_directory is not None:
                    models[key] = model

            print(f"- training: {timer.dt:.01f} s ... ", end="")

            with Timer() as timer:

                # Run the model report and save the it as a json file
                # Need the deepcopy here, still not entirely sure why
                report = deepcopy(Report())
                report.populate_performance(
                    model, x_train, y_train, x_test, y_test
                )
                if compute_feature_importance:
                    report.populate_feature_importance(
                        model, x_test, y_test, n_jobs
                    )

            print(f"- report/save: {timer.dt:.01f} s")

            self._reports[key] = report

            if debug > 0:
                if ii >= debug:
                    warn("In testing mode- ending early!")
                    break

        if output_data_directory is not None:
            Path(output_data_directory).mkdir(exist_ok=True, parents=True)
            root = Path(output_data_directory)
            report_path = root / f"{base_name}.json"
            with open(report_path, "w") as f:
                json.dump(self.to_json(), f, indent=4)

            # Save the model itself
            model_path = root / f"{base_name}_models.pkl"
            pickle.dump(
                models,
                open(model_path, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

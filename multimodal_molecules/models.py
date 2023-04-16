from functools import cache
import json
from pathlib import Path
import pickle
from time import perf_counter
from warnings import warn

from monty.json import MSONable
import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multimodal_molecules import DEFAULT_RANDOM_STATE


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def dt(self):
        return self._time


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

    def __init__(
        self,
        min_fg_occurrence=0.02,
        max_fg_occurrence=0.98,
        data_size=None,
        data_loaded_from=None,
        report=None,
        functional_groups_subset=None,
        pca_components_per_modality=0,
        nmf_components_per_modality=0,
        other_rf_kwargs={},
        n_estimators_per_modality=100,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        self._min_fg_occurrence = min_fg_occurrence
        self._max_fg_occurrence = max_fg_occurrence
        self._data_size = data_size
        self._data_loaded_from = data_loaded_from
        self._functional_groups_subset = functional_groups_subset

        if pca_components_per_modality > 0 and nmf_components_per_modality > 0:
            raise ValueError("Choose one PCA or NMF, not both")

        self._pca_components_per_modality = pca_components_per_modality
        self._nmf_components_per_modality = nmf_components_per_modality

        # There was a reason for doing this but I can't remember what it was
        # I.e. I don't recommend just setting report={} as a kwarg default...
        if report is None:
            self._report = {}
        else:
            self._report = report

        self._other_rf_kwargs = other_rf_kwargs
        self._n_estimators_per_modality = n_estimators_per_modality
        self._random_state = random_state

    def run_experiments(
        self,
        data,
        output_data_directory=None,
        n_jobs=2,
        debug=-1,
        compute_feature_importance=False,
    ):
        """Runs all experiments corresponding to the functional groups
        and the initially provided conditions.

        Parameters
        ----------
        data : multimodal_molecules.data.XANESData
            The data class.
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

        root = Path(output_data_directory)

        functional_groups = data.available_functional_groups
        if self._functional_groups_subset is not None:
            functional_groups = self._functional_groups_subset

        all_binary_targets = data.get_FG_data()

        for jj, combo in enumerate(data.available_combinations):
            X_train = data.get_XANES_data(combo, index_subset="train")
            X_valid = data.get_XANES_data(combo, index_subset="valid")

            print(
                f"\nCurrent XANES combo={combo} - "
                f"X_train.shape={X_train.shape}, "
                f"X_valid.shape={X_valid.shape}"
            )

            for ii, fg_name in enumerate(functional_groups):
                # Print some diagonstic information...
                ename = f"{'_'.join(combo)}-{fg_name}"
                print(f"\n\t[{(ii+1):03}/{len(functional_groups):03}] {ename}")

                model_path = root / f"{ename}_model.pkl"
                if Path(model_path).exists():
                    print(f"\tmodel={model_path} exists, continuing")
                    continue

                # Check that the occurence of the functional groups falls into
                # the specified sweet spot
                b = all_binary_targets[fg_name]
                p = b.sum() / len(b)
                if not self._min_fg_occurrence < p < self._max_fg_occurrence:
                    print(
                        f"{SPACING}Occ. {p:.04f} - continuing\n",
                        flush=True,
                    )
                    continue

                y_train = data.get_FG_data(fg_name, index_subset="train")
                y_valid = data.get_FG_data(fg_name, index_subset="valid")

                p_val = y_valid.sum() / len(y_valid)
                p_train = y_train.sum() / len(y_train)

                if jj == 0:
                    print(
                        f"{SPACING}"
                        f"Occ. total={p:.04f} | train={p_train:.04f} | "
                        f"val={p_val:.04f} "
                    )

                # Train the model
                with Timer() as timer:
                    Lc = len(combo)

                    n_estimators = self._n_estimators_per_modality * Lc
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                        random_state=self._random_state,
                        **self._other_rf_kwargs,
                    )

                    if self._pca_components_per_modality > 0:
                        model = Pipeline(
                            steps=[
                                ("scaler", StandardScaler()),
                                (
                                    "pca",
                                    PCA(
                                        self._pca_components_per_modality * Lc
                                    ),
                                ),
                                ("rf", model),
                            ]
                        )

                    elif self._nmf_components_per_modality > 0:
                        model = Pipeline(
                            steps=[
                                (
                                    "nmf",
                                    NMF(
                                        self._nmf_components_per_modality * Lc
                                    ),
                                ),
                                ("rf", model),
                            ]
                        )

                    print(
                        f"{SPACING}Fitting model with "
                        f"n_estimators={n_estimators}"
                    )
                    model.fit(X_train, y_train)
                    assert len(model.estimators_) == n_estimators

                print(f"{SPACING}Training: {timer.dt:.01f} s")

                with Timer() as timer:
                    y_val_pred = model.predict(X_valid)
                    y_train_pred = model.predict(X_train)

                    # Accuracies and other stuff
                    self._report[ename] = {
                        "p_total": p,
                        "p_train": p_train,
                        "p_val": p_val,
                        "val_accuracy": accuracy_score(y_valid, y_val_pred),
                        "train_accuracy": accuracy_score(
                            y_train, y_train_pred
                        ),
                        "val_balanced_accuracy": balanced_accuracy_score(
                            y_valid, y_val_pred
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
                            model, X_valid, y_valid, n_jobs=n_jobs
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
            base_name = data.conditions.replace(",", "_")
            report_path = root / f"{base_name}.json"
            with open(report_path, "w") as f:
                json.dump(self.to_json(), f, indent=4, sort_keys=True)
            print(f"\nReport saved to {report_path}")

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def save_json(d, path):
    with open(path, "w") as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def read_json(path):
    with open(path, "r") as infile:
        dat = json.load(infile)
    return dat


def scaler_from_estimator(est):
    scaler = StandardScaler()
    scaler.fit(est.X_train)
    d_scaler = dict(vars(scaler))
    for key, value in d_scaler.items():
        if isinstance(value, np.ndarray):
            d_scaler[key] = list(value)
        elif isinstance(value, np.int64):
            d_scaler[key] = int(value)
    return d_scaler, scaler


def _torch_models_from_Crescendo(target, ensemble):
    """Converts crescendo (6fcfcb7e44e0c96ab9dd23a4f90755dcc7eb2ac7) models
    into pure pytorch models so they can be easily used.

    This is a utility that basically nobody except me (Matt, the author) will
    need. Please feel free to ignore this.

    Parameters
    ----------
    target : os.PathLike
        The root target directory where everything should be saved.
    ensemble : Crescendo Ensemble
        Note that this object also points to the location of the data used
        during training and whatnot.

    Example
    -------
    element = "O"
    xanes_dir = f"{element}-XANES"
    ensemble = Ensemble.from_root(
        f"data/23-05-05-ensembles/{element}",
        data_dir=f"data/23-04-26-ml-data/{xanes_dir}"
    )
    root = Path(f"data/23-12-06_torch_models/{element}")
    """

    for ii, est in enumerate(ensemble.estimators):

        target_directory = target / f"{ii:02}"
        target_directory.mkdir(exist_ok=True, parents=True)

        # Get the model itself...
        model = est.get_model()

        # Need to also get the specific scaling information
        # we turn the scaler into a dictionary to serialize it and
        # avoid pickle when possible
        d_scaler, scaler = scaler_from_estimator(est)
        save_json(d_scaler, target_directory / "scaler.json")
        X = scaler.transform(est.X_train)
        del scaler

        # Check right here that this works
        scaler2 = StandardScaler()
        scaler2_dict = read_json(target_directory / "scaler.json")
        for key, value in scaler2_dict.items():
            setattr(scaler2, key, value)
        X2 = scaler2.transform(est.X_train)
        del scaler2

        assert np.all(X == X2)

        # Save the model itself as a standalone tensor
        # We'll use checksums for validation later
        torch.save(model, target_directory / "model.pt")

        with torch.no_grad():
            model.eval()
            pred1 = model(torch.FloatTensor(X)).detach().numpy()
        del model

        # Check by reading and asserting things are the same
        model2 = torch.load(target_directory / "model.pt")

        # And assert everything is the same
        with torch.no_grad():
            model2.eval()
            pred2 = model2(torch.FloatTensor(X2)).detach().numpy()
        del model2

        assert np.all(pred1 == pred2)


class Estimator:
    @classmethod
    def from_path(cls, path):
        path = Path(path)
        with open(path / "scaler.json", "r") as infile:
            scaler_dict = json.load(infile)
        scaler = StandardScaler()
        for key, value in scaler_dict.items():
            setattr(scaler, key, value)
        model = torch.load(path / "model.pt")
        return cls(model, scaler)

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).detach().numpy()


class Ensemble:
    @classmethod
    def from_path(cls, path):
        estimators = []
        for path2 in Path(path).iterdir():
            if not path2.is_dir():
                continue
            if not (path2 / "model.pt").exists():
                continue
            estimators.append(Estimator.from_path(path2))
        return cls(estimators)

    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        return np.array([est.predict(X) for est in self.estimators])


def _read_text_file(fname):
    with open(fname, "r") as f:
        return [line.strip() for line in f.readlines()]


def get_data(target="data/23-04-26-ml-data", elements="C"):
    """Gets the data corresponding to the provided elements. Helper function
    to retrieve the data for validation and further testing.

    Parameters
    ----------
    target : os.PathLike, optional
        The root path to where the data is located.
    elements : str, optional
        Either "C", "N", "O", "CN", "CO", "ON", "CNO".

    Returns
    -------
    dict
        A dictionary containing all of the information you could ever want
        about the data.
    """

    xanes_string = "_".join([f"{el}-XANES" for el in sorted(elements)])
    d = Path(target) / xanes_string
    return {
        "functional_groups": _read_text_file(d / "functional_groups.txt"),
        "smiles_test": _read_text_file(d / "smiles_test.txt"),
        "smiles_val": _read_text_file(d / "smiles_val.txt"),
        "smiles_train": _read_text_file(d / "smiles_train.txt"),
        "X_train": np.load(d / "X_train.npy"),
        "X_val": np.load(d / "X_val.npy"),
        "X_test": np.load(d / "X_test.npy"),
        "Y_train": np.load(d / "Y_train.npy"),
        "Y_val": np.load(d / "Y_val.npy"),
        "Y_test": np.load(d / "Y_test.npy"),
    }

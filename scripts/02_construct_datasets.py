from pathlib import Path

import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split

from multimodal_molecules.data import get_dataset
from crescendo.preprocess.array import ensemble_split


def screening(fg_dict, low=0.05, high=0.95):
    """Screen out functional groups that don't appear in the data
    very often."""

    new_d = {}
    for key, value in fg_dict.items():
        avg = value.mean()
        if low < avg < high:
            new_d[key] = value
    return new_d


def concatenate(d):
    keys = [key for key in d.keys() if "XANES" in key]
    keys.sort()
    X = np.concatenate([d[key] for key in keys], axis=1)
    return X


conditions = [
    "C-XANES",
    "N-XANES",
    "O-XANES",
    "C-XANES,N-XANES",
    "C-XANES,O-XANES",
    "N-XANES,O-XANES",
    "C-XANES,N-XANES,O-XANES",
]


def construct_standard_data_splits():
    for condition in conditions:
        d = Path("data") / "23-05-03-ml-data" / condition.replace(",", "_")
        d.mkdir(exist_ok=True, parents=True)

        data = get_dataset(
            xanes_path=Path("data") / "22-12-05-data" / "xanes.pkl",
            index_path=Path("data") / "22-12-05-data" / "index.csv",
            conditions=condition,
        )
        X = concatenate(data)
        assert X.shape[1] == len(condition.split(",")) * 200

        new_fg = screening(data["FG"])
        columns = list(new_fg.keys())
        Y = np.array([v for v in new_fg.values()]).T

        (
            X_train_val,
            X_test,
            y_train_val,
            y_test,
            smiles_train_val,
            smiles_test,
        ) = train_test_split(
            X, Y, data["index"]["SMILES"], test_size=0.15, random_state=42
        )
        (
            X_train,
            X_val,
            y_train,
            y_val,
            smiles_train,
            smiles_val,
        ) = train_test_split(
            X_train_val,
            y_train_val,
            smiles_train_val,
            test_size=0.15,
            random_state=42,
        )

        assert (
            X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == X.shape[1]
        )
        assert (
            y_train.shape[1]
            == y_val.shape[1]
            == y_test.shape[1]
            == Y.shape[1]
            == len(columns)
        )

        np.save(d / "X_train.npy", X_train)
        np.save(d / "X_val.npy", X_val)
        np.save(d / "X_test.npy", X_test)

        np.save(d / "Y_train.npy", y_train)
        np.save(d / "Y_val.npy", y_val)
        np.save(d / "Y_test.npy", y_test)

        with open(d / "smiles_train.txt", "w") as f:
            for line in smiles_train.to_list():
                f.write(f"{line}\n")

        with open(d / "smiles_val.txt", "w") as f:
            for line in smiles_val.to_list():
                f.write(f"{line}\n")

        with open(d / "smiles_test.txt", "w") as f:
            for line in smiles_test.to_list():
                f.write(f"{line}\n")

        with open(d / "functional_groups.txt", "w") as f:
            for line in columns:
                f.write(f"{line}\n")

        print(
            X_train.shape,
            X_val.shape,
            X_test.shape,
            y_train.shape,
            y_val.shape,
            y_test.shape,
        )


def get_num_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return len(mol.GetAtoms())


def construct_cutoff_data_splits(cutoff=8):
    for condition in conditions:
        d = (
            Path("data")
            / f"23-05-11-ml-data-CUTOFF{cutoff}"
            / condition.replace(",", "_")
        )
        d.mkdir(exist_ok=True, parents=True)

        data = get_dataset(
            xanes_path=Path("data") / "22-12-05-data" / "xanes.pkl",
            index_path=Path("data") / "22-12-05-data" / "index.csv",
            conditions=condition,
        )
        data["index"]["num_atoms"] = data["index"]["SMILES"].apply(
            get_num_atoms
        )

        X = concatenate(data)
        assert X.shape[1] == len(condition.split(",")) * 200

        new_fg = screening(data["FG"])
        columns = list(new_fg.keys())
        Y = np.array([v for v in new_fg.values()]).T

        print(condition, X.shape, Y.shape)

        where_train = np.where(data["index"]["num_atoms"] <= cutoff)[0]
        where_test = np.where(data["index"]["num_atoms"] > cutoff)[0]

        smiles_train_val = data["index"]["SMILES"].iloc[where_train]
        smiles_test = data["index"]["SMILES"].iloc[where_test]

        X_train_val = X[where_train, :]
        X_test = X[where_test, :]
        y_train_val = Y[where_train, :]
        y_test = Y[where_test, :]

        (
            X_train,
            X_val,
            y_train,
            y_val,
            smiles_train,
            smiles_val,
        ) = train_test_split(
            X_train_val,
            y_train_val,
            smiles_train_val,
            test_size=0.15,
            random_state=42,
        )

        assert (
            X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == X.shape[1]
        )
        assert (
            y_train.shape[1]
            == y_val.shape[1]
            == y_test.shape[1]
            == Y.shape[1]
            == len(columns)
        )

        np.save(d / "X_train.npy", X_train)
        np.save(d / "X_val.npy", X_val)
        np.save(d / "X_test.npy", X_test)

        np.save(d / "Y_train.npy", y_train)
        np.save(d / "Y_val.npy", y_val)
        np.save(d / "Y_test.npy", y_test)

        with open(d / "smiles_train.txt", "w") as f:
            for line in smiles_train.to_list():
                f.write(f"{line}\n")

        with open(d / "smiles_val.txt", "w") as f:
            for line in smiles_val.to_list():
                f.write(f"{line}\n")

        with open(d / "smiles_test.txt", "w") as f:
            for line in smiles_test.to_list():
                f.write(f"{line}\n")

        with open(d / "functional_groups.txt", "w") as f:
            for line in columns:
                f.write(f"{line}\n")

        ensemble_split(d, n_splits=20, shuffle=True, random_state=42)

        print(
            X_train.shape,
            X_val.shape,
            X_test.shape,
            y_train.shape,
            y_val.shape,
            y_test.shape,
            len(smiles_train),
            len(smiles_val),
            len(smiles_test),
        )

        y_train_mean = y_train.mean(axis=0)
        print(f"{y_train_mean.min():.02f} -> {y_train_mean.max():.02f}")

        y_val_mean = y_val.mean(axis=0)
        print(f"{y_val_mean.min():.02f} -> {y_val_mean.max():.02f}")
        print("---" * 8)


if __name__ == "__main__":
    construct_standard_data_splits()
    construct_cutoff_data_splits(cutoff=8)

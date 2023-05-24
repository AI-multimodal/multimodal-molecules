import json
from pathlib import Path
import pickle

import pandas as pd


def read_json(path):
    with open(path, "r") as infile:
        dat = json.load(infile)
    return dat


d = Path("data") / "22-12-05-data"


if __name__ == "__main__":
    data = pickle.load(open(d / "xanes.pkl", "rb"))
    functional_group_data = read_json(d / "functional_groups.json")
    all_functional_groups_enumerated = [
        g for groups in functional_group_data.values() for g in groups
    ]
    all_unique_functional_groups = sorted(
        list(set(all_functional_groups_enumerated))
    )

    index = {
        "SMILES": [],
        "C": [],
        "N": [],
        "O": [],
        "C-XANES": [],
        "N-XANES": [],
        "O-XANES": [],
    }
    index = {**index, **{fg: [] for fg in all_unique_functional_groups}}

    for smile, dat in data["data"].items():
        lower_smile = smile.lower()

        index["SMILES"].append(smile)

        for key in ["C", "N", "O"]:
            index[key].append(int(key.lower() in lower_smile))

        for key in ["C-XANES", "N-XANES", "O-XANES"]:
            index[key].append(int(dat[key] is not None))

        for fg in all_unique_functional_groups:
            index[fg].append(int(fg in functional_group_data[smile]))

    df = pd.DataFrame(index)
    df.to_csv(d / "index.csv")
    index = pd.read_csv(d / "index.csv", index_col=0)  # Reload
    assert (df == index).all().all()

from functools import lru_cache

import json
import numpy as np
import pandas as pd
import pickle


def save_json(d, path):
    with open(path, "w") as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def read_json(path):
    with open(path, "r") as infile:
        dat = json.load(infile)
    return dat


def load_pickle(path):
    return pickle.load(open(path, "rb"))


@lru_cache
def get_pickle_data_and_cache(path):
    return load_pickle(path)


@lru_cache
def get_json_data_and_cache(path):
    return read_json(path)


@lru_cache
def get_csv_data_and_cache(path):
    return pd.read_csv(path, index_col=0)


@lru_cache
def get_dataset(xanes_path, index_path, conditions="C-XANES"):
    """Summary

    Parameters
    ----------
    xanes_path : os.PathLike
        Points to a pickle file, which corresponds to the dictionary of data
        with SMILES strings as keys and arbitrary data as value, but this data
        must contain "X-XANES" keys, where X == C, N, O. Note that the pickle
        file itself must have the key "data", which will be what is accessed.
    index_path : TYPE
        Description
    conditions : str, optional
        Description

    Returns
    -------
    dict
    """

    conditions = conditions.split(",")

    # Complementary conditions
    # e.g. X-XANES => X but X does NOT => X-XANES
    cc_conditions = [xx.split("-")[0] for xx in conditions if "XANES" in xx]
    conditions = conditions + cc_conditions
    print(f"Getting data, applying conditions: {conditions}")

    xanes = get_pickle_data_and_cache(xanes_path)
    grids = xanes["grids"]
    index = get_csv_data_and_cache(index_path)

    # Refine the index dataframe until all conditions have been applied
    # This is a bit of a hack but I think it'd be harder/more confusing and
    # not much faster to try and stack "&" conditions...
    for condition in conditions:
        if "!" != condition[0]:
            index = index[index[condition] == 1]
        else:
            index = index[index[condition[1:]] == 0]

    # The resulting index contains all SMILES we want and in the right order
    smiles = index["SMILES"].to_list()

    xanes_conditions = [cc for cc in conditions if "XANES" in cc]
    final_data = {
        key: np.array([xanes["data"][smile][key] for smile in smiles])
        for key in xanes_conditions
    }
    final_data["grid"] = {
        key: grids[key.split("-")[0]] for key in xanes_conditions
    }

    # Finally, get all functional groups
    # The functional groups begin being referenced at row #7
    fg_index = index.iloc[:, 7:]
    final_data["FG"] = dict()
    for fg in fg_index.columns:
        dat = fg_index[fg].to_numpy()
        if dat.sum() > 0:
            final_data["FG"][fg] = dat

    final_data["index"] = index

    return final_data

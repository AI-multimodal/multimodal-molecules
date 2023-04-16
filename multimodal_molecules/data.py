from functools import cached_property, cache
from itertools import combinations
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pickle

from multimodal_molecules import DEFAULT_RANDOM_STATE


def load_pickle(path):
    return pickle.load(open(path, "rb"))


@cache
def get_pickle_data_and_cache(path):
    return load_pickle(path)


@cache
def get_csv_data_and_cache(path):
    return pd.read_csv(path, index_col=0)


@cache
def get_dataset(xanes_path, index_path, conditions="C-XANES"):
    """Summary

    Parameters
    ----------
    xanes_path : os.PathLike
        Points to a pickle file, which corresponds to the dictionary of data
        with SMILES strings as keys and arbitrary data as value, but this data
        must contain "X-XANES" keys, where X == C, N, O. Note that the pickle
        file itself must have the key "data", which will be what is accessed.
    index_path : os.PathLike
        Points to the csv index file generated in a previous step.
    conditions : str, optional
        A query that specifies which kinds of data to take in the subset. Some
        examples of reasonable queries are:

        .. code::

            # All molecules containing at least one C and one O XANES
            >> "C-XANES,O-XANES"

            # All molecules containing at least one C XANEs but no nitrogen
            >> "C-XANES,!N"

    Returns
    -------
    dict
        A dictionary containing keys like ['C-XANES', 'O-XANES', 'grid', 'FG',
        'index'], where the '*-XANES' keys are the XANES spectra, the 'grid' is
        the energy grid, 'FG' is a dictionary containing the functional groups
        that are present at least once in the data selected. 'index' is the
        index file itself that is required for this function.
    """

    conditions = conditions.split(",")

    # Complementary conditions
    # e.g. X-XANES => X but X does NOT => X-XANES
    cc_conditions = [xx.split("-")[0] for xx in conditions if "XANES" in xx]
    conditions = conditions + cc_conditions
    print(f"Getting data, applying conditions: {conditions}")
    print(f"Loading xanes_path={xanes_path}")
    print(f"Loading index_path={index_path}")

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


def get_all_combinations(elements):
    """Gets all possible combinations given the list elements.

    Parameters
    ----------
    elements : list

    Returns
    -------
    list
    """

    combos = []
    for nn in range(len(elements)):
        combos.extend(list(combinations(elements, nn + 1)))
    return combos


class XANESData:
    """Container for managing the data."""

    @cached_property
    def data_dict(self):
        """Gets the original data dictionary as produced by get_dataset. This
        is a cached property, so it will only load/process once.

        Returns
        -------
        dict
        """

        return get_dataset(self.xanes_path, self.index_path, self._conditions)

    @cached_property
    def available_modalities(self):
        """The XANES modalities present.

        Returns
        -------
        list
        """

        return [ee for ee in self._conditions.split(",") if "XANES" in ee]

    @cached_property
    def single_spectrum_length(self):
        """The length of a single modality's spectrum. Note this assumes
        that all modalities are of the same length and will throw an assertion
        error if this is not the case.

        Returns
        -------
        int
        """

        data_subset = [self.data_dict[kk] for kk in self.available_modalities]
        dlen = [xx.shape[1] for xx in data_subset.values()]
        assert len(set(dlen)) == 1
        return dlen[0] // self.available_modalities

    @cached_property
    def total_datapoints(self):
        return self.data_dict[self.available_modalities[0]].shape[0]

    @cached_property
    def available_combinations(self):
        return get_all_combinations(self.available_modalities)

    def get_XANES_data(self, modalities=None, index_subset=None):
        """Gets the XANES data resolved by the list of provided modalities.

        Parameters
        ----------
        modalities : list
            A list like ["C-XANES", "O-XANES"]. If modalities is None, returns
            everything.
        index_subset : array_like or str, optional
            Either an array specifying the indexes to return, or one of
            {"train", "valid", "test""}.

        Returns
        -------
        numpy.ndarray
        """

        if modalities is not None:
            modalities = sorted(modalities)
        else:
            modalities = sorted(self.available_modalities)

        o1 = self._offset_left
        o2 = self._offset_right
        X = np.concatenate(
            [self.data_dict[key][:, o1:o2] for key in modalities],
            axis=1,
        )

        if index_subset is None:
            return X
        elif isinstance(index_subset, (np.ndarray, list)):
            return X[index_subset, :]
        elif isinstance(index_subset, str):
            # Train, valid or test...
            idx_train, idx_valid, idx_test = self.train_valid_test_indexes
            if index_subset == "train":
                return X[idx_train, :]
            elif index_subset == "valid":
                return X[idx_valid, :]
            elif index_subset == "test":
                return X[idx_test, :]
            else:
                raise ValueError(f"Unknown indexes {index_subset}")

        raise RuntimeError("Unknown error")

    @cached_property
    def train_valid_test_indexes(self):
        """Gets three lists of indexes corresponding to the training,
        validation and testing splits. Note this property is cached.

        Returns
        -------
        list, list, list
        """

        random.seed(self._random_state)
        indexes = [ii for ii in range(self.total_datapoints)]
        random.shuffle(indexes)
        test_size = int(self._test_size * self.total_datapoints)
        val_size = int(self._val_size * self.total_datapoints)
        t_plus_v = test_size + val_size

        test_indexes = indexes[:test_size]
        val_indexes = indexes[test_size:t_plus_v]
        train_indexes = indexes[t_plus_v:]

        assert set(train_indexes).isdisjoint(set(val_indexes))
        assert set(train_indexes).isdisjoint(set(test_indexes))
        assert set(test_indexes).isdisjoint(set(val_indexes))

        return sorted(train_indexes), sorted(val_indexes), sorted(test_indexes)

    @cached_property
    def available_functional_groups(self):
        """A list of the available functional groups in this dataset.

        Returns
        -------
        list
        """

        return list(self.data_dict["FG"].keys())

    def get_FG_data(self, fg_list=None, index_subset=None):
        """Returns a dictionary keyed by the functional group and containing
        the binary representation of whether or not at least one functional
        group is present in the molecule or not.

        Parameters
        ----------
        fg_list : list or str, optional
            If not provided, uses all functional groups.
        index_subset : array_like or str, optional
            Either an array specifying the indexes to return, or one of
            {"train", "valid", "test""}.

        Returns
        -------
        dict or numpy.ndarray

        Raises
        ------
        KeyError
            If the provided list of functional groups is not a subset of those
            available.
        RuntimeError
            Description
        ValueError
            Description
        """

        if fg_list is None:
            grps = self.data_dict["FG"]

        else:
            try:
                if isinstance(fg_list, str):
                    grps = self.data_dict["FG"][fg_list]
                elif len(fg_list) == 1:
                    grps = self.data_dict["FG"][fg_list[0]]
                else:
                    grps = {key: self.data_dict["FG"][key] for key in fg_list}

            except KeyError as err:
                avail = self.available_functional_groups
                print("Available functional groups:", avail)
                raise KeyError(err)

        if index_subset is None:
            return grps
        elif isinstance(index_subset, (np.ndarray, list)):
            return grps[index_subset]
        elif isinstance(index_subset, str):
            # Train, valid or test...
            idx_train, idx_valid, idx_test = self.train_valid_test_indexes
            if index_subset == "train":
                return grps[idx_train]
            elif index_subset == "valid":
                return grps[idx_valid]
            elif index_subset == "test":
                return grps[idx_test]
            else:
                raise ValueError(f"Unknown indexes {index_subset}")

        raise RuntimeError("Unknown error")

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, *args, **kwargs):
        """Only set the random state at instantiation!!!"""

        raise ValueError("Common now, you trying to break the code?")

    @property
    def conditions(self):
        return self._conditions

    @property
    def xanes_path(self):
        return str(Path(self._data_directory) / "xanes.pkl")

    @property
    def index_path(self):
        return str(Path(self._data_directory) / "index.csv")

    def __init__(
        self,
        data_directory,
        conditions="C-XANES,N-XANES,O-XANES",
        random_state=DEFAULT_RANDOM_STATE,
        offset_left=None,
        offset_right=None,
        test_size=0.1,
        val_size=0.1,
    ):
        self._data_directory = data_directory
        self._conditions = ",".join(sorted(conditions.split(",")))
        self._random_state = random_state
        self._offset_left = offset_left
        self._offset_right = offset_right
        self._test_size = test_size
        self._val_size = val_size

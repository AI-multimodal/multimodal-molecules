from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from crescendo.analysis import HPTunedSet


def multimodal_errors(
    data_name,
    data_dir,
    modalities,
    ensemble_path,
    save_path,
):
    with open(Path(data_dir) / "functional_groups.txt", "r") as f:
        functional_groups = f.readlines()
    functional_groups = [xx.strip() for xx in functional_groups]

    results_list = modalities.split(",")
    errors = {}
    Y_test = None

    for key in results_list:
        p = Path(ensemble_path) / key

        hptuned = HPTunedSet.from_root(p, data_dir=data_dir)
        best_estimator, best_value = hptuned.get_best_estimator(
            hptuned.X_val, hptuned.Y_val
        )
        result = best_estimator.predict(hptuned.X_test)
        if Y_test is None:
            Y_test = hptuned.Y_test
        else:
            assert np.allclose(Y_test, hptuned.Y_test)
        errors[key] = [
            balanced_accuracy_score(result[:, ii].round(), Y_test[:, ii])
            for ii in range(len(functional_groups))
        ]
        del hptuned
        del best_estimator
        del result

    df = pd.DataFrame({"functional_group": functional_groups})
    for key, value in errors.items():
        df[key] = value

    df.to_csv(save_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    multimodal_errors(*args)

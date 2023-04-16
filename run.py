from pprint import pprint
import sys

import json

CONDITIONS = [
    "C-XANES",
    "N-XANES",
    "O-XANES",
    "C-XANES,N-XANES",
    "C-XANES,O-XANES",
    "N-XANES,O-XANES",
    "C-XANES,N-XANES,O-XANES",
]


if __name__ == "__main__":

    with open("config.json", "r") as f:
        config = json.load(f)

    print("Loaded config...")
    pprint(config)

    sys.path.append(config["module_path"])
    from multimodal_molecules.models import Results
    from multimodal_molecules.data import XANESData

    for condition in CONDITIONS:

        data = XANESData(conditions=condition, **config["XANESData_kwargs"])
        results = Results(**config["Results_kwargs"])
        results.run_experiments(
            data,
            output_data_directory=condition.replace(",", "_"),
            **config["run_experiments_kwargs"]
        )

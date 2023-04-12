import sys

import json

if __name__ == "__main__":

    with open("config.json", "r") as f:
        config = json.load(f)

    sys.path.append(config["module_path"])
    from multimodal_molecules.models import Results

    results = Results(**config["results_kwargs"])
    results.run_experiments(**config["run_experiments_kwargs"])

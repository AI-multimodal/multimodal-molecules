from copy import deepcopy

from multimodal_molecules.models import Results

OUTPUT_DIR = "results/221223"
INPUT_DIR = "data/221205"
CONDITIONS = [
    "C-XANES",
    "N-XANES",
    "O-XANES",
    "C-XANES,N-XANES",
    "C-XANES,O-XANES",
    "C-XANES,N-XANES,O-XANES",
]

if __name__ == "__main__":
    for cc in CONDITIONS:
        results = deepcopy(
            Results(
                conditions=cc,
                xanes_data_name="221205_xanes.pkl",
                index_data_name="221205_index.csv",
            )
        )
        results.run_experiments(
            input_data_directory=INPUT_DIR,
            output_data_directory=OUTPUT_DIR,
            n_jobs=4,
            debug=-1,
            compute_feature_importance=True,
        )

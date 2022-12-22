from multimodal_molecules.models import Results

OUTPUT_DIR = "results/221222"
CONDITIONS = [
    "O-XANES",
    "N-XANES",
    "O-XANES,N-XANES",
]

if __name__ == "__main__":
    for cc in CONDITIONS:
        results = Results(
            conditions=cc,
            xanes_data_name="221205_xanes.pkl",
            index_data_name="221205_index.csv",
        )
        results.run_experiments(
            input_data_directory="data/221205",
            output_data_directory="results/221222",
            n_jobs=8,
            debug=-1,
            compute_feature_importance=True,
        )

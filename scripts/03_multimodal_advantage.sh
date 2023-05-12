#!/bin/bash -l

ml_data_path=data/23-04-26-ml-data
ensemble_path=data/23-05-03-hp
save_dir=data/23-05-12-multimodal-single-estimator

data="C-XANES_N-XANES"
modalities="C-N_only_C,C-N_only_N,C-N"
python3 ../multimodal_molecules/postprocessing/multimodal_advantage.py \
    "$data" \
    "$ml_data_path"/"$data" \
    "$modalities" \
    "$ensemble_path" \
    "$save_dir"/"$data".csv

data="C-XANES_O-XANES"
modalities="C-O_only_C,C-O_only_O,C-O"
python3 multimodal_molecules/postprocessing/multimodal_advantage.py \
    "$data" \
    "$ml_data_path"/"$data" \
    "$modalities" \
    "$ensemble_path" \
    "$save_dir"/"$data".csv


data="N-XANES_O-XANES"
modalities="N-O_only_N,N-O_only_O,N-O"
python3 multimodal_molecules/postprocessing/multimodal_advantage.py \
    "$data" \
    "$ml_data_path"/"$data" \
    "$modalities" \
    "$ensemble_path" \
    "$save_dir"/"$data".csv


data="C-XANES_N-XANES_O-XANES"
modalities="C-N-O_only_C,C-N-O_only_N,C-N-O_only_O,C-N-O"
python3 multimodal_molecules/postprocessing/multimodal_advantage.py \
    "$data" \
    "$ml_data_path"/"$data" \
    "$modalities" \
    "$ensemble_path" \
    "$save_dir"/"$data".csv

ml_data_path=data/23-05-11-ml-data-CUTOFF8
ensemble_path=data/23-05-12-hp-CUTOFF8
data="C-XANES_N-XANES_O-XANES"
modalities="C-N-O_only_C,C-N-O_only_N,C-N-O_only_O,C-N-O"
python3 multimodal_molecules/postprocessing/multimodal_advantage.py \
    "$data" \
    "$ml_data_path"/"$data" \
    "$modalities" \
    "$ensemble_path" \
    "$save_dir"/"$data"-CUTOFF8.csv

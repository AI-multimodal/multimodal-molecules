# Data

- `22-12-05-data`: original molecular XANES data created from [Ghose _et al._](https://doi.org/10.1103/PhysRevResearch.5.013180).
- `23-04-26-ml-data`: machine learning-ready data which is prepared in the format required by [Crescendo](https://github.com/matthewcarbone/Crescendo).
- `23-05-03-multimodal-molecules-hp-tuning`: hyper-parameter tuning results from `23-04-26-ml-data`.
- `23-05-05-ensembles`: ensemble results from `23-04-26-ml-data`.
- `23-05-10-ml-data-train-on-small`: a special machine learning-ready dataset constructed by a unique partitioning: only molecules with `<=7` atoms/molecule are used for training/validation, the rest are used for testing. The splits can be found below.

| Type | Total  | FG  | N train | N val | N test |
| ---- | ------ | --- | ------- | ----- | ------ |
| C    | 130453 | 36  | 3327    | 588   | 126538 |
| N    | 82275  | 42  | 1860    | 329   | 80086  |
| O    | 113085 | 36  | 2641    | 467   | 109977 |
| CNO  | 65382  | 42  | 1350    | 239   | 63793  | 

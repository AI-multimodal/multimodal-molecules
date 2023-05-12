# Data

- `22-12-05-data`: original molecular XANES data created from [Ghose _et al._](https://doi.org/10.1103/PhysRevResearch.5.013180).
- `23-04-26-ml-data`: machine learning-ready data which is prepared in the format required by [Crescendo](https://github.com/matthewcarbone/Crescendo).
- `23-05-03-hp`: hyper-parameter tuning results from `23-04-26-ml-data`.
- `23-05-05-ensembles`: ensemble results from `23-04-26-ml-data`.
- `23-05-11-ml-data-CUTOFF8`: a special machine learning-ready dataset constructed by a unique partitioning: only molecules with `<=8` atoms/molecule are used for training/validation, the rest are used for testing. The splits can be found below.

| Type | Total  | FG  | N train | N val | N test |
| ---- | ------ | --- | ------- | ----- | ------ |
| C    | 130453 | 36  | 18530   | 3271  | 108652 |
| N    | 82275  | 42  | 11137   | 1966  | 69172  |
| O    | 113085 | 36  | 15453   | 2728  | 94904  |
| CNO  | 65382  | 42  | 8516    | 1503  | 55363  |

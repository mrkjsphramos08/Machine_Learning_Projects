# Notebooks — the "bare ML" workbench for this project

This folder is where you actually do Machine Learning: exploring a dataset,
plotting distributions, testing feature ideas, prototyping models.

Suggested flow:

1. `01_eda.ipynb` — load `data/raw/` and *learn the data*: shape, dtypes,
   missing values, target distribution, leakage risks, outliers.
2. `02_features.ipynb` — test feature engineering ideas and the validation
   split you intend to use.
3. `03_models.ipynb` — compare candidate algorithms **on the same split with
   the same metric** before committing to one.

## When to graduate out of the notebook

Move code into `../src/` as soon as an idea works and you want it to run again
on new data. Notebooks are excellent for exploration but poor for:

- reproducible pipeline stages (`dvc.yaml`)
- unit / smoke testing (`pytest`)
- reuse across projects

The conversion is mechanical: copy the working cell into a function in
`../src/data.py` (data work) or `../src/train.py` (model work), give it type
hints, then delete the scratch cell.

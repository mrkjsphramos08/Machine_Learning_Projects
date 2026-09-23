---
name: dataset-acquisition
description: Procedures for obtaining datasets from Kaggle, public URLs or scikit-learn, recording provenance and licensing, placing files in a project's data/raw, versioning them with DVC, and avoiding the leakage traps public datasets ship with.
---

# Dataset Acquisition Skill

Use this when bringing a new dataset into a project. Getting the data is easy;
getting it *reproducibly* and *without built-in leakage* is the part that needs a
procedure.

---

## 1. Kaggle datasets (recommended: `kagglehub`)

No API token is needed for public datasets.

```powershell
pip install -r requirements-dev.txt          # includes kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('uciml/iris'))"
```

`dataset_download` prints the path of a **cache directory** — copy the file you
need into the project, do not point your config at the cache:

```powershell
Copy-Item "$env:USERPROFILE\.cache\kagglehub\datasets\uciml\iris\versions\1\Iris.csv" `
          "projects\02_kaggle_iris\data\raw\iris.csv"
```

Hedge: `kagglehub` also exposes a competition download (e.g.
`kagglehub.competition_download('titanic')`). Competition files require accepting
the rules on the Kaggle website first. Run `dir(kagglehub)` after installing to
confirm the available functions in your version.

### Kaggle CLI alternative

```powershell
pip install kaggle
# Place your token at %USERPROFILE%\.kaggle\kaggle.json (Account > Create New API Token)
kaggle datasets download -d uciml/iris -p data/raw --unzip
```

---

## 2. Public URLs, UCI, or a built-in dataset

```python
# One-off download, then save it into the project so DVC can track a stable file
import pandas as pd
df = pd.read_csv("https://example.com/data.csv")
df.to_csv("data/raw/data.csv", index=False)
```

Built-ins (fastest way to make progress when the real dataset is not ready):

```python
from sklearn.datasets import load_diabetes, fetch_california_housing
X, y = fetch_california_housing(as_frame=True, return_X_y=True)
```

Whatever the source, the rule is the same: **one immutable file in
`data/raw/`**, and nothing downstream ever re-downloads it.

---

## 3. Record provenance (do this immediately, it is unrecoverable later)

Write `data/raw/DATASET.md` next to the file:

```markdown
# Dataset provenance
- Source URL : https://www.kaggle.com/datasets/uciml/iris
- Retrieved  : 2026-09-23
- Licence    : CC0 Public Domain
- Rows/cols  : 150 / 5
- sha256     : <output of Get-FileHash below>
- Notes      : no missing values; 3 balanced classes
```

```powershell
Get-FileHash data\raw\iris.csv -Algorithm SHA256 | Select-Object -ExpandProperty Hash
```

The hash is what lets you prove later that the file DVC points at is the file you
actually analysed.

---

## 4. Version it with DVC

```powershell
cd projects\02_kaggle_iris
dvc add data/raw/iris.csv
cd ..\..
git add projects/02_kaggle_iris/data/raw/iris.csv.dvc projects/02_kaggle_iris/data/raw/DATASET.md
git commit -m "feat(data): track iris raw dataset (v1)"
```

Never commit CSVs, Parquet files or archives. `.gitignore` already blocks
`projects/*/data/raw/*` with an exception for `*.dvc` pointers.

If the dataset is shared across projects, consider a DVC remote instead of
copying it into both (see the `dvc-data-versioning` skill).

---

## 5. Leakage traps that ship with public datasets

Check every one of these **before** you trust your first score:

| Trap | How to detect |
| :--- | :--- |
| File already split (Kaggle `train.csv` / `test.csv`) | If you re-split `train.csv` you are fine; using Kaggle's `test.csv` for evaluation often means **no labels** |
| Duplicate rows across your own split | `df.duplicated().sum()` and check overlap of `train`/`test` keys |
| ID-like columns | `range(n)` shaped columns, `PassengerId`, `index` — drop them or they memorize |
| Target leakage columns | Any column populated *after* the outcome (e.g. `churn_reason`, `exit_date`) |
| Time leakage | Shuffling time-ordered data across the split boundary (see `validation-strategy`) |
| Group leakage | The same patient/customer appears in train **and** test |
| Pre-normalised features | Scaling computed on the whole dataset *before* splitting |

Rule of thumb: if a feature's value could not have been known at prediction time,
it must not be in `X`.

---

## 6. Checklist

- [ ] Exactly one immutable file in `data/raw/`, named in `configs/config.yaml`.
- [ ] `data/raw/DATASET.md` records URL, retrieval date, licence and sha256.
- [ ] `dvc add` run from the project folder; the `.dvc` pointer committed to Git.
- [ ] No CSV/Parquet/archive staged in Git (`git status` confirms).
- [ ] `configs/config.yaml` → `dataset.target_column` set to the real label.
- [ ] Leakage traps above reviewed and any suspect columns explicitly dropped.

---
name: exploratory-data-analysis
description: Notebook procedure for exploring a new dataset - first-pass profiling, target analysis by task type, feature distributions, integrity and leakage checks, useful plots, and the rule for graduating notebook code into src modules.
---

# Exploratory Data Analysis Skill

Use this the moment a dataset lands in `data/raw/`, **before** any modelling. The
output is not a pretty chart — it is a list of decisions: what to drop, what to
impute, what the split must be, and which metric matters.

---

## 0. Environment notes for this repo (verified against the installed versions)

- **pandas 3.0**: Copy-on-Write is permanently enabled. Chained assignment
  (`df["a"][0] = 1`) now raises `ChainedAssignmentError` and changes nothing —
  always use `df.loc[0, "a"] = 1`.
- **pandas 3.0**: strings are typed `str`, not `object`. `select_dtypes(include="object")`
  still finds them for now but emits a `Pandas4Warning` (it is deprecated), so
  pass `include="str"` explicitly — or `exclude="number"`, which catches string,
  categorical and boolean columns in one shot.
- `matplotlib` is installed; `seaborn` is optional
  (`pip install -r requirements-dev.txt`). Prefer matplotlib so the notebook runs
  on a bare environment. Note this goes against most Google results.
- Notebooks live in `projects/<name>/notebooks/`; kernel = the repo `.venv`.

---

## 1. First pass (~15 lines)

```python
import pandas as pd

df = pd.read_csv("data/raw/train.csv")     # relative to the PROJECT folder
df.shape
df.head()
df.dtypes
df.isna().sum().sort_values(ascending=False).head(20)   # missingness ranking
df.duplicated().sum()                                   # exact duplicate rows
df.nunique().sort_values()                              # cardinality -> ID columns

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()   # pandas 3 safe
```

Decisions to record: which columns are unusable, which are categorical, which
need imputing.

---

## 2. Target analysis — different per task

**Regression target**
```python
df[y].describe()                       # range, mean vs median (skew signal)
df[y].skew()                           # > 1 suggests a log transform
(df[y] == 0).mean()                    # zero-inflation -> maybe two-part model
```
Look for: heavy skew, impossible values (negative ages), a spike at a single
value, and outliers that will dominate RMSE.

**Classification target**
```python
df[y].value_counts(normalize=True)     # class balance - decides your metric
```
Look for: imbalance (decides accuracy vs F1/PR-AUC), tiny minority classes,
and label noise. A 95/5 split means accuracy is worthless — see
`metrics-and-evaluation`.

---

## 3. Feature analysis

| Question | Check |
| :--- | :--- |
| Which numeric features relate to the target? | `df[num_cols + [y]].corr(numeric_only=True)[y].sort_values()` |
| Do categorical levels differ in outcome? | `df.groupby(col)[y].agg(["mean", "count"])` |
| Is a "numeric" column really categorical? | `df[col].nunique()` small + integer dtype |
| Any constant or near-constant columns? | `df.nunique() == 1` |
| Datetime coverage and gaps? | `df[col].min()`, `.max()`, `df[col].diff().value_counts()` |
| Are there sentinel values for missing? | `-1`, `999`, `"unknown"`, empty strings (not counted by `isna()`!) |

That last row causes more silent bugs than any other: `-1` and `"unknown"` are
missing values that `isna()` cannot see.

---

## 4. Integrity and leakage checks

Run these explicitly, and cross-reference the trap table in the
`dataset-acquisition` skill:

```python
df.duplicated(subset=[id_col]).sum()          # duplicate entities
df.groupby(id_col).size().describe()          # repeated measures -> GroupKFold
df[date_col].is_monotonic_increasing          # time ordering -> TimeSeriesSplit
for col in cat_cols:                          # suspiciously perfect predictors
    print(col, df.groupby(col)[y].mean().round(3).to_dict())
```

If a feature separates the target almost perfectly, suspect leakage before you
celebrate.

---

## 5. Plots worth making (and only these)

1. Target distribution — histogram (regression) or bar of class counts.
2. Missingness — `df.isna().mean().sort_values().plot.barh()`.
3. Each numeric feature vs target — scatter, or boxplot by class.
4. Correlation heatmap of numeric features (spot redundant features).
5. Time series of the target if a date column exists.

Skip decorative plot work. Charts exist here to change a decision.

---

## 6. Graduate into `src/` (the important step)

When a finding becomes part of the pipeline, it moves out of the notebook:

| Finding in the notebook | Goes to |
| :--- | :--- |
| "column `x` is an ID, drop it" | `features.drop` in `configs/config.yaml` |
| "drop rows with literal `'unknown'` target" | `clean_data()` in `src/data.py` |
| "median-impute the numeric columns" | `build_preprocessor()` in `src/train.py` |
| "must split by patient group" | the splitter in the `prepare` stage |

If you cannot yet write it as a function with type hints, the idea is not ready.

---

## 7. EDA exit checklist

- [ ] Row/column counts and target definition written into the project README.
- [ ] Missing values identified, including sentinel-encoded ones.
- [ ] Categorical vs numeric columns declared in the config.
- [ ] Leakage suspects named and dropped.
- [ ] Split protocol chosen (see `validation-strategy`) based on duplicates/groups/time.
- [ ] Metric chosen based on the target distribution (see `metrics-and-evaluation`).

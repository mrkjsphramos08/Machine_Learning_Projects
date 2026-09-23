---
name: reproducibility-and-environments
description: How to make a run reproducible and how to reproduce an old one - the Git/DVC/MLflow tripod, seeds and determinism, dependency pinning, known non-determinism sources, and the current gaps in this repo.
---

# Reproducibility & Environments Skill

Use this when a result must be trustworthy, when weeks have passed, or when you
need to rebuild an old model. "It worked on my machine yesterday" is not
reproducibility.

---

## 1. The tripod: three systems, three jobs

| Leg | Pins | Location |
| :--- | :--- | :--- |
| **Git** | code, config, `dvc.yaml`/`dvc.lock` (hashes of data and outputs) | this repo |
| **DVC** | the dataset bytes and pipeline outputs | `.dvc/cache` (+ remote if configured) |
| **MLflow** | the run record: params, metrics, artifacts, source commit | `mlflow.db` + `mlruns/` |

You need all three. A commit without the data is a guess; a run without its commit
is an anecdote.

---

## 2. Reproduce a past run

```powershell
# 1. Which code? MLflow records the commit for runs it can attribute:
#    run tag mlflow.source.git.commit   (and mlflow.source.git.branch)
# 2. Pin it
git checkout <commit>
dvc checkout                 # materialise the exact data/outputs from the cache
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd projects\<name>
..\..\.venv\Scripts\python.exe -m src.train
```

Then compare the metrics with the run you were resurrecting. If they differ, the
difference is a reproducibility bug worth chasing (see §4).

---

## 3. Current gaps in this repo (verified)

| Gap | Consequence | Remedy |
| :--- | :--- | :--- |
| **No DVC remote** (`dvc remote list` is empty) | tracked data exists only in `.dvc/cache` on this disk; `dvc push` is impossible and a disk failure loses it | `dvc remote add -d storage <path-or-s3-url>` then `dvc push` |
| Only some runs carry a git commit tag | you cannot always trace a metric back to code | log it explicitly (§5) |
| `requirements.txt` uses `>=`, not `==` | a reinstall months later may resolve different versions | keep a committed lock file (§6) |
| Non-windows platforms | line-ending and floating-point differences | state the platform in the run description |

The missing DVC remote is the biggest one: it is the difference between
"versioned" and "versioned on one disk".

---

## 4. Sources of non-determinism

| Source | Mitigation |
| :--- | :--- |
| Random splitting/model init | `random_state` everywhere (already done in `configs/config.yaml`) |
| Sampling, shuffling, bootstrapping | seed the generator explicitly |
| `n_jobs=-1` parallelism | small float differences; use `n_jobs=1` when bitwise equality matters |
| Floating point across CPUs/BLAS | record the platform; do not promise bitwise equality in general |
| Set/dict iteration order | sort before iterating over collections |
| File glob order | `sorted(Path(...).glob(...))` |
| Parquet row order | sort explicitly if downstream output depends on order |
| GPU kernels | generally not bitwise reproducible; document it |

Practical rule: **reproducible to the metric, not necessarily to the last decimal.**

---

## 5. Log the commit yourself

MLflow records `mlflow.source.git.commit` only when it can attribute the run, and
that has not been every run in this repo. Assert it explicitly:

```python
import subprocess
import mlflow


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("git_commit", git_commit())
    mlflow.set_tag("data_version", "see dvc.lock")
```

`dvc.lock` is the other half: commit it together with the code so the data hash and
the code hash move in lockstep.

---

## 6. Dependency pinning

| Approach | Reproducibility |
| :--- | :--- |
| `scikit-learn>=1.3.0` | loose — resolves differently over time |
| `scikit-learn==1.9.1` | exact, but you must bump deliberately |
| committed lock file | best of both: ranges in `requirements.txt`, exact pins in the lock |

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```

MLflow already snapshots each logged model's environment (`conda.yaml`,
`python_env.yaml`, `requirements.txt`) inside the run's artifact directory — useful
for tracing which versions produced a given model.

---

## 7. Pre-flight check before trusting a number

- [ ] `git status` clean (or the dirty files noted).
- [ ] `dvc status` reports up to date.
- [ ] `random_state` set for split **and** model, and logged in the run.
- [ ] Two consecutive runs produce identical metrics.
- [ ] The run carries a commit tag (`mlflow.source.git.commit`) or `git_commit`.
- [ ] Package versions recorded (lock file or the MLflow environment artifact).
- [ ] The metric is compared against the dummy baseline and the frozen holdout.

---
name: ml-cicd
description: Setting up continuous integration for this ML repo with GitHub Actions - caching, running pytest and dvc repro without credentials, building the Docker image, secrets handling, and PR gates.
---

# ML CI/CD Skill

Use this when you have a git remote and want every push checked without thinking
about it. **Precondition:** this repo currently has no git remote and no DVC
remote (`git remote -v` and `dvc remote list` are both empty) — see §2 before
writing any workflow.

---

## 1. What CI should run, in order

1. install pinned dependencies
2. lint/format check (optional: `ruff`)
3. `pytest` — the fast suite only
4. `dvc repro -P` **or** `dvc pull` — reproduce or fetch data, never re-download from Kaggle
5. `docker build` on the main branch only (see `docker-ml-packaging`)

Cheap checks first: a failing test should not wait for a Docker build.

---

## 2. Prerequisites for data in CI

CI runners have no DVC cache. Pick one:

| Option | Trade-off |
| :--- | :--- |
| DVC remote (S3/GCS/Azure/local SSH) | the correct answer; needs credentials in CI secrets |
| A tiny committed fixture dataset for tests | keeps tests hermetic; no credentials needed |
| Re-download from Kaggle in CI | needs `KAGGLE_USERNAME`/`KAGGLE_KEY` secrets and is slow/flaky |

Recommended split: **tests use synthetic fixtures** (they already do), and CI runs
`dvc repro` only if a DVC remote exists. That keeps the pipeline credential-free.

---

## 3. GitHub Actions workflow

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip                       # caches wheels between runs

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: pytest -q -m "not slow"       # env is hermetic; no data needed

      - name: Reproduce pipelines
        if: ${{ hashFiles('.dvc/config') != '' }}
        run: |
          pip install -r requirements.txt
          dvc pull || echo "no DVC remote configured - skipping pull"
          dvc repro -P

  image:
    needs: test
    if: github.ref == 'refs/heads/master'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build serving image
        run: docker build -t mlops-service:${{ github.sha }} .
```

Notes that save time:
- The `-m "not slow"` filter keeps PR feedback under a minute (declare the marker
  in `pytest.ini` — see `ml-testing-strategy`).
- `cache: pip` on `setup-python` is the single highest-value caching win.
- Do not start `mlflow ui` in CI; runs go to a throwaway store inside the runner.
- Pin `actions/*` major versions to avoid surprise breakage.

---

## 4. Secrets

- Never commit `kaggle.json`, tokens or service-account files (already covered by
  `.gitignore` patterns for `*.db` and `.venv`, but Kaggle tokens are **not**
  matched — keep them in `%USERPROFILE%\.kaggle\` and out of the repo).
- Store CI credentials under **Settings → Secrets and variables → Actions**.
- Prefer short-lived, scoped credentials (a read-only object-store key) over
  long-lived personal tokens.

---

## 5. Gates worth enforcing

| Gate | Why |
| :--- | :--- |
| `pytest` green | the baseline contract in `.agents/rules/mlops_standards.md §4` |
| `dvc repro -P` clean | the pipeline still reproduces from raw data |
| `git status` clean after repro | catches generated files that were not committed |
| Image builds | packaging does not rot |

A useful extra check for this repo specifically:

```yaml
      - name: Pipelines reproduce cleanly
        run: |
          dvc repro -P
          git diff --exit-code -- '**/dvc.lock' || echo "dvc.lock changed - commit it"
```

---

## 6. What CI should *not* do

- Train large models on every PR (use a tiny fixture dataset).
- Publish to a model registry from a PR (promotion is a deliberate act — see
  `model-promotion-and-aliases`).
- Run Kubernetes/cloud deploys before the tests are trustworthy.
- Store datasets or model artifacts as build artefacts.

---

## 7. Checklist

- [ ] Git remote exists and CI runs on push and PR.
- [ ] Python version in CI matches `.venv` (3.11).
- [ ] Tests run with no credentials and no real dataset.
- [ ] `pip` cache enabled; runtime under ~2 minutes for the test job.
- [ ] `docker build` gated to the main branch.
- [ ] Slow tests excluded by marker, not by commenting them out.

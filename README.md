# Decision-model

Fine-tunes [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it) as a calibrated probabilistic decision model
on the [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark
(`all` config: 1,200 train cases, 400 test cases / 2,000 decisions), then publishes the model and its scores to the
Hugging Face Hub so they appear on the benchmark leaderboard.

Each (case, question) pair becomes one prompt listing the options as letters; the softmax of the next-token logits over
those letters is the answer distribution, trained with soft cross-entropy against the gold distribution.

## Files

| File | What it is |
|---|---|
| `gemma3_270m_invoice_decisions.ipynb` | The notebook: data, training, evaluation, inference examples, Hub publishing |
| `build_notebook.py` | Generates the notebook; edit this, then rebuild (see below) |
| `compare_runs.py` | Compares the saved runs in `runs/` (used by the notebook, also a CLI) |
| `runs/` | One folder per training run with its summary and test results (created by the notebook) |
| `pyproject.toml` / `uv.lock` | Dependencies for uv (locked versions) |
| `requirements.txt` | The same locked versions for plain pip, exported from `uv.lock` |
| `.python-version` | Python 3.12 |

## Setup with uv

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
Developed on Apple Silicon (PyTorch's MPS backend); it also runs on CPU, or on CUDA if you change `DEVICE`.

```bash
uv sync                      # creates .venv with Python 3.12 and the locked dependencies
uv run hf auth login         # token with write access (needed for the gated base model and for pushing)
```

`google/gemma-3-270m-it` is gated: accept its license on the model page with the same account first.

Run the notebook:

```bash
uv run jupyter lab gemma3_270m_invoice_decisions.ipynb
```

In VS Code / Cursor, pick `.venv/bin/python` as the notebook kernel instead.

Or run it headless end to end (outputs are written back into the notebook):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 gemma3_270m_invoice_decisions.ipynb
```

### Without uv

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Optional: hyperparameter search (Optuna, ≤ 2 h on a GPU)

Section 5a runs an Optuna search over learning rate, weight decay, warmup, gradient accumulation and epochs,
minimising validation KL. It is off by default; in the hyperparameter cell set:

```python
RUN_HPO          = True   # run the search
HPO_BUDGET_MIN   = 115    # hard wall-clock cap for the whole search (minutes)
HPO_TRAIN_FRAC   = 0.5    # each trial trains on this fraction of the training cases
USE_BEST_HPARAMS = True   # final training (5b) uses hpo/best_hparams.json if present
```

* The cap is enforced inside the training loop too, so a running trial is cut off at the deadline
  (Optuna's own `timeout` only stops new trials from starting). Poor trials are pruned after each epoch.
* The first trial is the default hyperparameters, so the result can only match or beat them.
* Trials are stored in `hpo/optuna.db`, so re-running continues the study. The winner is written to
  `hpo/best_hparams.json`. You can run the search on a GPU machine, copy that file back, and train anywhere.
  With `RUN_HPO = False` the final training still picks the file up.
* Device is picked automatically (CUDA → MPS → CPU); on CUDA, training and inference use bf16 autocast.
* On Linux, `uv sync` installs the CUDA build of PyTorch from PyPI.

## Run tracking and comparing runs

Every time the notebook trains and evaluates a model (section 6b) it saves a summary to
`runs/<timestamp>_<model>_<config>[_<RUN_TAG>]/`:

| File | Contents |
|---|---|
| `summary.json` | model, config, device, git commit, dataset revision, hyperparameters and their source (defaults / Optuna), data sizes, per-epoch training curve, test metrics overall / by type / by workflow / by question, baselines |
| `summary.md` | the same as a readable report |
| `train_history.csv`, `training_curves.png` | per-epoch train loss and validation metrics |
| `test_by_task.csv`, `test_by_type.csv`, `test_by_workflow.csv`, `comparison.csv` | test tables, incl. zero-shot, Uniform and Prior baselines |
| `test_predictions.jsonl` | full predicted distributions for every test case |
| `hpo_trials.csv` | all Optuna trials, if the search ran in that session |

Set `RUN_TAG` in the hyperparameter cell to label a run. `runs/index.csv` (one row per run) is rebuilt after each
run. The files are small, so commit them to keep the history.

Compare runs in section 10 of the notebook, or from a terminal:

```bash
uv run python compare_runs.py                              # one row per run, sorted by test KL
uv run python compare_runs.py --sort test_acc              # sorted by test accuracy
uv run python compare_runs.py --by type                    # test accuracy per question type, one column per run
uv run python compare_runs.py --by workflow --metric kl    # test KL per workflow
uv run python compare_runs.py --by task --runs "runs/*hpo*"   # only some runs (glob)
```

## Editing the notebook

The notebook is generated from `build_notebook.py`, so make changes there and rebuild (this overwrites the notebook
and clears its outputs):

```bash
uv run python build_notebook.py
```

## Managing dependencies

```bash
uv add <package>             # add a dependency and update uv.lock
uv lock --upgrade            # bump everything within the constraints in pyproject.toml
uv export --no-hashes --no-header --no-annotate --format requirements-txt -o requirements.txt   # refresh requirements.txt
```

## Publishing to the leaderboard

Section 9 of the notebook writes the model card, `test_predictions.jsonl`, `test_metrics.json` and
`.eval_results/typed-decisions.yaml` (the [Hub eval-results](https://huggingface.co/docs/hub/en/eval-results) format)
into the output folder. Nothing is uploaded until you set `PUSH_TO_HUB = True` (model repo, picked up by the
benchmark's leaderboard) and `OPEN_DISCUSSION = True` (posts the scores on the dataset's discussion board for the
README tables). The model is fine-tuned on `train`, so it belongs to the "fitted or fine-tuned on `train`" table.

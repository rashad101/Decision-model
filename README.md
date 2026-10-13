# Decision-model

Fine-tunes small language models as calibrated probabilistic decision models on the
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark
(`all` config: 1,200 train cases, 400 test cases / 2,000 decisions), then publishes each model and its scores to the
Hugging Face Hub so they appear on the benchmark leaderboard.

Each (case, question) pair becomes one prompt listing the options as letters; the softmax of the next-token logits over
those letters is the answer distribution, trained with soft cross-entropy against the gold distribution.

Two models, one notebook each, sharing the same recipe:

| Notebook | Base model | Params | Fine-tune | Final epochs |
|---|---|---|---|---|
| `gemma3_270m_decisions.ipynb` | [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) (gated) | 270M | full, fp32 | `EPOCHS` (from the search or the default 12) |
| `ministral3_3b_reasoning_decisions.ipynb` | [`mistralai/Ministral-3-3B-Reasoning-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) (Apache-2.0) | 3.4B (text decoder) | full, fp32 weights + bf16 autocast, embeddings frozen | fixed **10** (`FINAL_EPOCHS`) |

Both pick the epoch with the lowest validation KL for the test evaluation.

## Files

| File | What it is |
|---|---|
| `gemma3_270m_decisions.ipynb` | Gemma 3 270M: data, training, evaluation, inference examples, Hub publishing |
| `ministral3_3b_reasoning_decisions.ipynb` | The same for Ministral 3 3B Reasoning, published as a **(Specialist)** model |
| `runs/` | One folder per training run with its summary and test results (created by the notebooks, shared by both models) |
| `hpo/` | Optuna searches: `hpo/` for Gemma, `hpo/ministral3-3b-reasoning/` for Ministral |
| `pyproject.toml` / `uv.lock` | Dependencies for uv (locked versions) |
| `requirements.txt` | The same locked versions for plain pip, exported from `uv.lock` |
| `.python-version` | Python 3.12 |

Not in git (see `.gitignore`): the model output folders (`gemma3-270m-typed-decisions/`,
`ministral3-3b-reasoning-typed-decisions/`; weights go to the Hub), `checkpoints/` (Ministral's best-epoch weights
during training, ~14 GB) and the Optuna databases.

## Setup with uv

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
The device is picked automatically (CUDA → MPS → CPU); on CUDA, training and inference use bf16 autocast. On Linux,
`uv sync` installs the CUDA build of PyTorch from PyPI.

```bash
uv sync                      # creates .venv with Python 3.12 and the locked dependencies
uv run hf auth login         # token with write access (needed for the gated Gemma model and for pushing)
```

`google/gemma-3-270m-it` is gated: accept its license on the model page with the same account first. The Ministral
model is not gated.

Run a notebook:

```bash
uv run jupyter lab gemma3_270m_decisions.ipynb
uv run jupyter lab ministral3_3b_reasoning_decisions.ipynb
```

In VS Code / Cursor, pick `.venv/bin/python` as the notebook kernel instead.

Or run one headless end to end (outputs are written back into the notebook):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 ministral3_3b_reasoning_decisions.ipynb
```

### Without uv

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Hardware

* **Gemma 3 270M:** full fine-tuning in fp32 needs ~5 GB; fine on Apple Silicon (MPS) or any recent GPU.
  ~5.5 min per epoch on an NVIDIA GB10.
* **Ministral 3 3B Reasoning:** written for a CUDA GPU with lots of memory (developed on an NVIDIA GB10, 128 GB
  unified memory). Full fine-tuning peaks at ~51 GB with `BATCH_SIZE = 4`; use `BATCH_SIZE = 2, GRAD_ACCUM = 8` if
  memory is short. On a GB10 the GPU shares system memory, so shut down other notebook kernels first. Expect up to
  ~1 h per epoch, i.e. most of a day for the 10-epoch final run plus the search.

## Ministral 3 3B Reasoning specifics

* The checkpoint is an image-text model (`Mistral3ForConditionalGeneration`: Pixtral vision tower + text decoder).
  The task is text-only, so the notebook keeps just the decoder and LM head as a `Ministral3ForCausalLM`.
* **Reasoning is switched off.** The chat template's default system prompt asks the model to think in a
  `[THINK]…[/THINK]` block first. The notebook passes its own system prompt and pre-fills an empty
  `[THINK][/THINK]` after `[/INST]`, so the next token is the answer letter and one forward pass gives the distribution.
* The tokenizer is loaded with `fix_mistral_regex=True` (the shipped `tokenizer.json` has a known pre-tokenizer regex bug).
* The input embedding (tied to the LM head, 0.4B params) is frozen to save optimizer memory (3.0B trainable params).
  Option logits are computed only at the last position, from the LM-head rows of the option letters.
* The best epoch's weights are written to `checkpoints/ministral3-3b-reasoning_best.pt` instead of being kept in
  memory; the final model is saved in bf16.
* It is published as a **(Specialist)** model (fine-tuned for this benchmark only): the label is in the model card,
  the results row, the eval-results notes and the submission discussion title.

## Optional: hyperparameter search (Optuna)

Section 5a runs an Optuna search minimising validation KL. Set it in the hyperparameter cell:

```python
RUN_HPO          = True   # run the search
HPO_BUDGET_MIN   = 180    # hard wall-clock cap for the whole search (minutes); Gemma notebook default: 60
HPO_TRAIN_FRAC   = 0.25   # each trial trains on this fraction of the training cases; Gemma notebook default: 0.5
USE_BEST_HPARAMS = True   # final training (5b) uses <HPO_DIR>/best_hparams.json if present
```

| | Gemma 3 270M | Ministral 3 3B Reasoning |
|---|---|---|
| learning rate | 5e-6 – 2e-4 | 2e-6 – 5e-5 |
| gradient accumulation (batch 8 / 4) | 1, 2, 4 | 2, 4, 8 |
| epochs per trial | 2 – 5 | 1 – 3 (the final run always uses 10) |
| weight decay, warmup | 0 – 0.1, 0 – 0.2 | same |
| results | `hpo/` | `hpo/ministral3-3b-reasoning/` |

* The cap is enforced inside the training loop too, so a running trial is cut off at the deadline
  (Optuna's own `timeout` only stops new trials from starting). Poor trials are pruned after each epoch.
* The first trial is the default hyperparameters, so the result can only match or beat them.
* Trials are stored in `<HPO_DIR>/optuna.db`, so re-running continues the study. The winner is written to
  `<HPO_DIR>/best_hparams.json`. You can run the search on a GPU machine, copy that file back, and train anywhere.
  With `RUN_HPO = False` the final training still picks the file up.

## Run tracking and comparing runs

Every time a notebook trains and evaluates a model (section 6b) it saves a summary to
`runs/<timestamp>_<model>_<config>[_<RUN_TAG>]/`:

| File | Contents |
|---|---|
| `summary.json` | model, config, device, git commit, dataset revision, hyperparameters and their source (defaults / Optuna), data sizes, per-epoch training curve, test metrics overall / by type / by workflow / by question, baselines |
| `summary.md` | the same as a readable report |
| `train_history.csv`, `training_curves.png` | per-epoch train loss and validation metrics |
| `test_by_task.csv`, `test_by_type.csv`, `test_by_workflow.csv`, `comparison.csv` | test tables, incl. zero-shot, Uniform and Prior baselines |
| `test_predictions.jsonl` | full predicted distributions for every test case |
| `hpo_trials.csv` | all Optuna trials, if the search ran in that session |

Set `RUN_TAG` in the hyperparameter cell to label a run. `runs/index.csv` (one row per run, all models) is rebuilt
after each run. The files are small, so commit them to keep the history.

Compare runs in section 10 of a notebook: one row per run, then test metrics per question type / workflow and the
validation curves of every run, Gemma and Ministral side by side. The Ministral notebook defines the comparison helpers
(`load_runs`, `compare_by`, `pretty`) itself. The Gemma notebook still imports them from `compare_runs.py`, which has
been removed from the repo, so its sections 6b and 10 need those helpers copied over before they run.

## Managing dependencies

```bash
uv add <package>             # add a dependency and update uv.lock
uv lock --upgrade            # bump everything within the constraints in pyproject.toml
uv export --no-hashes --no-header --no-annotate --format requirements-txt -o requirements.txt   # refresh requirements.txt
```

## Publishing to the leaderboard

Section 9 of each notebook writes the model card, `test_predictions.jsonl`, `test_metrics.json` and
`.eval_results/typed-decisions.yaml` (the [Hub eval-results](https://huggingface.co/docs/hub/en/eval-results) format)
into the output folder. With `PUSH_TO_HUB = True` it uploads the model repo (picked up by the benchmark's
leaderboard), and with `OPEN_DISCUSSION = True` it posts the scores on the dataset's discussion board for the README
tables; both flags are on in the notebooks, so turn them off for runs you don't want published. The models are
fine-tuned on `train`, so they belong to the "fitted or fine-tuned on `train`" table; the Ministral submission is
titled `Submission: <user>/ministral3-3b-reasoning-typed-decisions (Specialist) (Ministral 3 3B Reasoning, fine-tuned on train)`.

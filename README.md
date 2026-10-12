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

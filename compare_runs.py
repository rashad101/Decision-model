"""Compare the runs saved by the notebook under runs/<run_id>/summary.json.

    uv run python compare_runs.py                    # one row per run: hyperparameters, training, test metrics
    uv run python compare_runs.py --by type          # test accuracy per question type, one column per run
    uv run python compare_runs.py --by workflow --metric kl
    uv run python compare_runs.py --by task --runs runs/20261012-*   # only some runs (glob)

Also rewrites runs/index.csv.
"""
import argparse
import glob
import json
import os

import pandas as pd


def _summaries(runs_dir="runs", pattern=None):
    paths = sorted(glob.glob(os.path.join(pattern or os.path.join(runs_dir, "*"), "summary.json")))
    return [json.load(open(p)) for p in paths]


def load_runs(runs_dir="runs", pattern=None):
    """One row per run with its setup, training result and overall test metrics."""
    rows = []
    for s in _summaries(runs_dir, pattern):
        hp, tr, t = s["hparams"], s["training"], s["test"]["overall"]
        rows.append({
            "run_id": s["run_id"], "created": s["created"], "tag": s.get("tag", ""),
            "model": s["model_id"].split("/")[-1], "config": s["config"], "device": s["device"],
            "hparams_from": "optuna" if s["hparams_source"] != "defaults" else "defaults",
            "lr": hp["lr"], "weight_decay": hp["weight_decay"], "warmup_frac": hp["warmup_frac"],
            "eff_batch": hp["effective_batch"], "epochs": hp["epochs"],
            "best_epoch": tr["best_epoch"], "val_kl": tr["best_val_kl"], "val_acc": tr["best_val_acc"],
            "train_min": tr["minutes"],
            "test_acc": t["accuracy"], "test_kl": t["kl"], "test_brier": t["brier"], "test_ece": t["ece"],
            "ms_per_case": s["test"]["ms_per_case"],
        })
    return pd.DataFrame(rows).set_index("run_id") if rows else pd.DataFrame()


def pretty(runs):
    """Round for display; the learning rate keeps scientific notation instead of rounding to 0."""
    out = runs.round(4)
    if "lr" in out:
        out["lr"] = runs["lr"].map("{:.1e}".format)
    return out


def compare_by(by="type", metric="accuracy", runs_dir="runs", pattern=None):
    """Rows = question types / workflows / questions, columns = runs, values = the chosen test metric."""
    cols = {s["run_id"]: {k: v[metric] for k, v in s["test"][f"by_{by}"].items()}
            for s in _summaries(runs_dir, pattern)}
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--runs", help="glob of run folders to include, e.g. 'runs/*hpo*'")
    ap.add_argument("--by", choices=["type", "workflow", "task"], help="break test metrics down per group")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "kl", "brier", "ece"])
    ap.add_argument("--sort", default="test_kl", help="column to sort the run table by")
    args = ap.parse_args()

    runs = load_runs(args.runs_dir, args.runs)
    if runs.empty:
        raise SystemExit(f"no runs found in {args.runs or args.runs_dir}")
    if args.runs is None:
        runs.to_csv(os.path.join(args.runs_dir, "index.csv"))
    pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60)
    if args.by:
        print(compare_by(args.by, args.metric, args.runs_dir, args.runs).round(4).to_string())
    else:
        ascending = not args.sort.endswith(("acc", "accuracy"))
        print(pretty(runs.sort_values(args.sort, ascending=ascending)).to_string())


if __name__ == "__main__":
    main()

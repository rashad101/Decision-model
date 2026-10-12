"""Builds gemma3_270m_invoice_decisions.ipynb. Run: .venv/bin/python build_notebook.py"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Fine-tuning Gemma 3 270M as a decision model — `typed-decisions / all`

Runs locally on Apple Silicon (MPS). Trains on the full `all` config (1,200 train cases, 4 workflows) and scores
the 400-case / 2,000-decision `test` split used by the leaderboard, then publishes the model and its scores to the Hub.

**Task.** Each case is one `state` plus 5 typed questions (`noul` yes/no, `choice`, `score` rubric). There are four workflows:

| workflow | questions |
|---|---|
| `agent_trace_observability` | action, needs_review, outcome, risk, urgency |
| `customer_service` | action, category, churn_risk, needs_human, urgency |
| `invoice_processing` | disposition, matches_order, duplicate, discrepancy_severity, urgency |
| `security_incidents` | disposition, true_positive, credential_compromise, severity, urgency |

Question names repeat across workflows (`urgency`, `disposition`, `action`), so results are keyed by `workflow/question`.

Gold answers are **probability distributions** (mean of 3 teacher samples), and the benchmark scores
accuracy *and* calibration (KL, Brier, ECE).

**Approach.** Each (case, question) pair becomes one prompt that lists the options as letters `A`, `B`, `C`, …
We read Gemma's next-token logits restricted to those letter tokens, softmax them into a distribution,
and fine-tune the full model with a **soft cross-entropy against the gold distribution**. This gives a proper
probabilistic classifier (no text generation or JSON parsing) and keeps the model's LM head as the decision head.
""")

md("## 1. Setup")
code("""
# Dependencies are in pyproject.toml / uv.lock: run `uv sync` in the repo and use .venv as the kernel.
# Without uv, uncomment:
# %pip install -q -r requirements.txt
""")
code("""
import json, math, random, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)
""")

md("""
### Hyperparameters
`google/gemma-3-270m-it` is gated: accept the license on its Hugging Face page and run `uv run hf auth login` once.
Full fine-tuning in fp32 needs ~5 GB of unified memory (weights + AdamW states), which fits an M3 Pro comfortably.
""")
code("""
MODEL_ID   = "google/gemma-3-270m-it"
CONFIG     = "all"    # all 4 workflows: 1,200 train / 400 test cases
MAX_LEN    = 1024     # longest prompt is ~800 tokens
BATCH_SIZE = 8
GRAD_ACCUM = 2        # effective batch 16
EPOCHS     = 4        # ~5,400 training decisions per epoch
LR         = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC  = 0.1
VAL_FRAC     = 0.1    # held-out part of train (by case) for model selection
OUT_DIR    = "gemma3-270m-typed-decisions"
""")

md("## 2. Data")
code("""
train_raw = load_dataset("LocalLLaMA/typed-decisions", CONFIG, split="train")
test_raw  = load_dataset("LocalLLaMA/typed-decisions", CONFIG, split="test")
print(train_raw, test_raw, sep="\\n")

row = train_raw[0]
state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
qname = next(iter(questions))
print(row["workflow"]); print(json.dumps(state, indent=1)[:1200])
print(json.dumps(questions[qname], indent=1))
print(json.dumps(gold[qname], indent=1))
print(pd.Series(train_raw["workflow"]).value_counts())
""")

md("""
### Turning questions into option lists
* `choice` → its `criteria` dict keys (label → description)
* `noul` → `true` / `false` (with descriptions when given; some, e.g. `duplicate`, have none, so we use a generic one)
* `score` → levels `0..N-1`, each with its rubric text

The gold `probabilities` dict is keyed by the same labels, so we can line them up as a vector.
""")
code("""
LETTERS = "ABCDEFGHIJ"

def question_options(q):
    \"\"\"Return [(label, description), ...] in a fixed order.\"\"\"
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        return list(crit.items())
    if t == "noul":
        crit = crit or {}
        return [("true",  crit.get("true",  "The statement is true.")),
                ("false", crit.get("false", "The statement is false."))]
    if t == "score":
        return [(str(i), d) for i, d in enumerate(crit)]
    raise ValueError(t)

SYSTEM = ("You are a precise decision model. Read the case, then answer the question "
          "with the single letter of the best option.")

def build_prompt(state, qname, q):
    opts = question_options(q)
    kind = {"choice": "Choose one option.",
            "noul":   "Is the statement true or false?",
            "score":  "Pick the level on the rubric."}[q["type"]]
    lines = [f"{LETTERS[i]}. {lab}: {desc}" for i, (lab, desc) in enumerate(opts)]
    return (f"{SYSTEM}\\n\\n### Case\\n{json.dumps(state, separators=(',', ':'))}\\n\\n"
            f"### Question ({qname})\\n{q['instructions']}\\n{kind}\\n\\n### Options\\n" + "\\n".join(lines)
            + "\\n\\nAnswer with one letter.")

def expand(ds, desc="expand"):
    \"\"\"One example per (case, question).\"\"\"
    out = []
    for r in tqdm(ds, desc=desc, unit="case"):
        state, qs, gold = (json.loads(r[k]) for k in ("state", "questions", "gold"))
        for qname, q in qs.items():
            opts = question_options(q)
            labels = [lab for lab, _ in opts]
            p = np.array([gold[qname]["probabilities"].get(l, 0.0) for l in labels], dtype=np.float32)
            p = p / p.sum()
            out.append(dict(case_id=r["id"], workflow=r["workflow"], qname=qname,
                            task=f"{r['workflow']}/{qname}", qtype=q["type"], labels=labels,
                            prompt=build_prompt(state, qname, q), gold_p=p,
                            gold_label=gold[qname]["label"]))
    return out

train_all = expand(train_raw, desc="build train prompts")
test_ex   = expand(test_raw,  desc="build test prompts")

# split train by case so no case leaks between train/val
case_ids = sorted({e["case_id"] for e in train_all})
random.Random(SEED).shuffle(case_ids)
val_cases = set(case_ids[:int(len(case_ids) * VAL_FRAC)])
train_ex = [e for e in train_all if e["case_id"] not in val_cases]
val_ex   = [e for e in train_all if e["case_id"] in val_cases]
print(f"train {len(train_ex)} | val {len(val_ex)} | test {len(test_ex)} examples")
print(train_ex[0]["prompt"])
""")

md("Label distribution of the gold argmax on train (useful to see class imbalance):")
code("""
pd.DataFrame(train_all).groupby("task")["gold_label"].value_counts(normalize=True).unstack().round(3)
""")

md("## 3. Model & tokenisation")
code("""
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(DEVICE)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# token id for each option letter (as the first token of the model's reply)
LETTER_IDS = []
for L in LETTERS:
    ids = tok.encode(L, add_special_tokens=False)
    assert len(ids) == 1, (L, ids)
    LETTER_IDS.append(ids[0])
print(dict(zip(LETTERS[:5], LETTER_IDS[:5])))

def to_chat(prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)

for name, split in (("train", train_ex), ("val", val_ex), ("test", test_ex)):
    for e in tqdm(split, desc=f"tokenize {name}", unit="ex"):
        e["input_ids"] = tok(to_chat(e["prompt"]), add_special_tokens=False)["input_ids"][:MAX_LEN]

lens = [len(e["input_ids"]) for e in train_ex]
print("prompt tokens: mean", int(np.mean(lens)), "max", max(lens))
""")

md("""
Batches are right-padded; we gather the logits at the **last real token** of each sequence
(the position after `<start_of_turn>model\\n`) and keep only the option-letter columns.
""")
code("""
def collate(batch):
    L = max(len(e["input_ids"]) for e in batch)
    pad = tok.pad_token_id
    ids  = torch.full((len(batch), L), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"]); mask[i, :n] = 1
    K = max(len(e["labels"]) for e in batch)
    target = torch.zeros((len(batch), K))
    opt_mask = torch.zeros((len(batch), K), dtype=torch.bool)
    for i, e in enumerate(batch):
        k = len(e["labels"])
        target[i, :k] = torch.from_numpy(e["gold_p"]); opt_mask[i, :k] = True
    return ids, mask, target, opt_mask

LETTER_IDS_T = torch.tensor(LETTER_IDS, device=DEVICE)   # sliced to the batch's option count (max 5 in `all`)

def option_logits(model, ids, mask, opt_mask):
    out = model(input_ids=ids, attention_mask=mask)
    last = mask.sum(1) - 1
    logits = out.logits[torch.arange(ids.size(0), device=ids.device), last]   # (B, V)
    logits = logits[:, LETTER_IDS_T[:opt_mask.size(1)]]                        # (B, K)
    return logits.masked_fill(~opt_mask, float("-inf"))

def soft_ce(logits, target):
    logp = F.log_softmax(logits.float(), dim=-1).masked_fill(target == 0, 0.0)
    return -(target * logp).sum(-1).mean()
""")

md("## 4. Evaluation helpers\n"
   "Metrics per decision, then averaged:\n"
   "* **Accuracy** – argmax of prediction equals the gold `label`\n"
   "* **KL(gold ‖ pred)** – how far the predicted distribution is from the gold one\n"
   "* **Brier** – Σₖ (pₖ − gₖ)² against the soft gold distribution\n"
   "* **ECE** – 10-bin expected calibration error of max-probability vs. correctness")
code("""
@torch.no_grad()
def predict(model, examples, bs=16, desc="predict"):
    model.eval()
    preds = []
    for i in tqdm(range(0, len(examples), bs), desc=desc, unit="batch", leave=False,
                  disable=len(examples) <= bs):
        batch = examples[i:i + bs]
        ids, mask, _, opt_mask = collate(batch)
        lg = option_logits(model, ids.to(DEVICE), mask.to(DEVICE), opt_mask.to(DEVICE))
        P = F.softmax(lg.float(), -1).cpu().numpy()
        for e, p in zip(batch, P):
            preds.append(p[:len(e["labels"])])
    return preds

def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return sum(np.abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
               for b in range(n_bins) if (bins == b).any())

def score(examples, preds, by="task"):
    \"\"\"Metrics grouped by `by` ("task", "workflow" or "qtype"), plus an OVERALL row.\"\"\"
    rows = []
    for e, p in tqdm(zip(examples, preds), total=len(examples), desc="score", unit="ex", leave=False):
        g = e["gold_p"]; pc = np.clip(p, 1e-8, 1)
        rows.append(dict(task=e["task"], workflow=e["workflow"], qtype=e["qtype"],
                         correct=e["labels"][int(p.argmax())] == e["gold_label"],
                         kl=float(np.sum(np.where(g > 0, g * (np.log(np.clip(g, 1e-8, 1)) - np.log(pc)), 0))),
                         brier=float(np.sum((p - g) ** 2)), conf=float(p.max())))
    df = pd.DataFrame(rows)
    summary = lambda d: pd.Series(dict(accuracy=d.correct.mean(), kl=d.kl.mean(), brier=d.brier.mean(),
                                       ece=ece(d.conf, d.correct), n=len(d)))
    per_q = df.groupby(by).apply(summary, include_groups=False)
    per_q.loc["OVERALL"] = summary(df)
    return per_q.round(4)
""")

md("### Reference baselines on the test split\n"
   "*Uniform* (equal probability on every option) and *Prior* (train label frequencies, ignoring the input).")
code("""
uniform_preds = [np.full(len(e["labels"]), 1 / len(e["labels"])) for e in test_ex]

prior = {}
for e in tqdm(train_all, desc="label priors", unit="ex"):
    prior.setdefault(e["task"], []).append(e["gold_p"])
prior = {q: np.mean(v, 0) for q, v in prior.items()}
prior_preds = [prior[e["task"]] for e in test_ex]

print("Uniform"); display(score(test_ex, uniform_preds).loc[["OVERALL"]])
print("Prior");   display(score(test_ex, prior_preds).loc[["OVERALL"]])
""")

md("### Zero-shot Gemma 3 270M (before fine-tuning)")
code("""
t0 = time.time()
zs_preds = predict(model, test_ex, desc="zero-shot test")
print(f"{time.time() - t0:.1f}s for {len(test_ex)} decisions")
zero_shot = score(test_ex, zs_preds)
zero_shot
""")

md("## 5. Fine-tuning\n"
   "Plain PyTorch loop: AdamW, cosine schedule with warmup, gradient accumulation, "
   "and selection of the epoch with the lowest validation KL (calibration is what the benchmark rewards).")
code("""
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = math.ceil(len(train_ex) / BATCH_SIZE / GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
sched = get_cosine_schedule_with_warmup(optim, int(WARMUP_FRAC * total_steps), total_steps)

best_kl, best_state, history = float("inf"), None, []
for epoch in tqdm(range(1, EPOCHS + 1), desc="epochs", unit="epoch"):
    model.train()
    order = list(range(len(train_ex)))
    random.shuffle(order)
    t0, run_loss, n = time.time(), 0.0, 0
    bar = tqdm(enumerate(range(0, len(order), BATCH_SIZE)), total=math.ceil(len(order) / BATCH_SIZE),
               desc=f"epoch {epoch}/{EPOCHS}", unit="batch", leave=False)
    for step, i in bar:
        ids, mask, target, opt_mask = collate([train_ex[j] for j in order[i:i + BATCH_SIZE]])
        lg = option_logits(model, ids.to(DEVICE), mask.to(DEVICE), opt_mask.to(DEVICE))
        loss = soft_ce(lg, target.to(DEVICE))
        (loss / GRAD_ACCUM).backward()
        run_loss += loss.item(); n += 1
        if (step + 1) % GRAD_ACCUM == 0 or i + BATCH_SIZE >= len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        bar.set_postfix(loss=f"{run_loss / n:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
    val = score(val_ex, predict(model, val_ex, desc=f"val epoch {epoch}")).loc["OVERALL"]
    history.append(dict(epoch=epoch, train_loss=run_loss / n, val_acc=val.accuracy, val_kl=val.kl,
                        val_brier=val.brier, minutes=(time.time() - t0) / 60))
    tqdm.write(str({k: round(v, 4) for k, v in history[-1].items()}))
    if val.kl < best_kl:
        best_kl = val.kl
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    if DEVICE == "mps":
        torch.mps.empty_cache()

pd.DataFrame(history).round(4)
""")

code("""
import matplotlib.pyplot as plt
h = pd.DataFrame(history)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(h.epoch, h.train_loss, marker="o", label="train soft-CE")
ax[0].plot(h.epoch, h.val_kl, marker="o", label="val KL")
ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("Loss")
ax[1].plot(h.epoch, h.val_acc, marker="o", color="tab:green")
ax[1].set_xlabel("epoch"); ax[1].set_title("Val accuracy")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
""")

md("## 6. Evaluate on the test split")
code("""
model.load_state_dict(best_state)
t0 = time.time()
ft_preds = predict(model, test_ex, desc="fine-tuned test")
ms_per_case = (time.time() - t0) / len(test_raw) * 1000
print(f"{ms_per_case:.0f} ms per case (5 decisions) on {DEVICE}")
fine_tuned = score(test_ex, ft_preds)
display(score(test_ex, ft_preds, by="qtype"))
display(score(test_ex, ft_preds, by="workflow"))
fine_tuned
""")

code("""
comparison = pd.DataFrame({
    "Uniform":            score(test_ex, uniform_preds).loc["OVERALL"],
    "Prior":              score(test_ex, prior_preds).loc["OVERALL"],
    "Gemma-270M zero-shot": zero_shot.loc["OVERALL"],
    "Gemma-270M fine-tuned": fine_tuned.loc["OVERALL"],
}).T.drop(columns="n")
comparison
""")

md("Per-question (`workflow/question`) accuracy, zero-shot vs fine-tuned:")
code("""
pd.DataFrame({"zero-shot": zero_shot.accuracy, "fine-tuned": fine_tuned.accuracy,
              "Δ": fine_tuned.accuracy - zero_shot.accuracy}).round(3)
""")

md("### Inspect a few test predictions")
code("""
by_case = {}
for e, p in zip(test_ex, ft_preds):
    by_case.setdefault(e["case_id"], []).append((e, p))

for cid in [next(c for c, v in by_case.items() if v[0][0]["workflow"] == w) for w in sorted({e["workflow"] for e in test_ex})]:
    print("=" * 80, "\\n", cid)
    for e, p in by_case[cid]:
        fmt = lambda v: ", ".join(f"{l}={x:.2f}" for l, x in zip(e["labels"], v))
        mark = "✓" if e["labels"][int(p.argmax())] == e["gold_label"] else "✗"
        print(f"{mark} {e['task']:<48} pred [{fmt(p)}]\\n  {'':<48} gold [{fmt(e['gold_p'])}]")
""")

md("## 7. Save the model and an inference helper")
code("""
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
json.dump({"letters": LETTERS, "letter_ids": LETTER_IDS, "system": SYSTEM}, open(f"{OUT_DIR}/decider_config.json", "w"))
print("saved to", OUT_DIR)
""")
code("""
@torch.no_grad()
def decide(state: dict, questions: dict) -> dict:
    \"\"\"Answer a typed-decisions request: returns {question: {label, confidence, probabilities}}.\"\"\"
    exs = []
    for qname, q in questions.items():
        labels = [l for l, _ in question_options(q)]
        exs.append(dict(qname=qname, labels=labels, gold_p=np.zeros(len(labels), np.float32),
                        input_ids=tok(to_chat(build_prompt(state, qname, q)), add_special_tokens=False)["input_ids"][:MAX_LEN]))
    out = {}
    for e, p in zip(exs, predict(model, exs)):
        probs = dict(zip(e["labels"], map(float, p)))
        out[e["qname"]] = dict(label=e["labels"][int(p.argmax())], confidence=float(p.max()), probabilities=probs)
    return out

def show(result):
    \"\"\"Pretty-print the output of decide().\"\"\"
    for qname, r in result.items():
        probs = "  ".join(f"{l}={p:.2f}" for l, p in r["probabilities"].items())
        print(f"{qname:<22} -> {r['label']:<14} ({r['confidence']:.0%})   [{probs}]")

r = test_raw[0]
print(r["workflow"])
show(decide(json.loads(r["state"]), json.loads(r["questions"])))
""")

md("""
## 8. Try it yourself
Reload the saved model from disk (so this section also works in a fresh session once sections 1–3 have run
and the helper cells above are defined), then run it on hand-written invoices. The five standard invoice questions
are taken from the dataset; edit `my_state` below, or paste any case JSON (from any workflow, with its questions),
to test your own input.
""")
code("""
model = AutoModelForCausalLM.from_pretrained(OUT_DIR, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(DEVICE).eval()
tok = AutoTokenizer.from_pretrained(OUT_DIR)

QUESTIONS = json.loads(next(r for r in test_raw if r["workflow"] == "invoice_processing")["questions"])
print(list(QUESTIONS))
""")

md("### Built-in examples")
code("""
def make_state(inv_id="INV-2026-5001", vendor="Acme Supplies", inv_qty=10, inv_unit=50.0,
               po_qty=10, po_unit=50.0, received_qty=10, condition="accepted",
               days_until_due=20, status="unpaid", prior_ids=(), disputes=0,
               discount_pct=None, discount_days=None):
    \"\"\"Build a case in the dataset's schema from a few knobs (single line item).\"\"\"
    return {
        "invoice": {"id": inv_id, "vendor": vendor, "currency": "USD",
                    "lines": [{"sku": "SKU-100", "qty": inv_qty, "unit_usd": inv_unit,
                               "total_usd": round(inv_qty * inv_unit, 2)}],
                    "total_usd": round(inv_qty * inv_unit, 2)},
        "purchase_order": {"id": "PO-9001", "lines": [{"qty": po_qty, "unit_usd": po_unit}],
                           "total_usd": round(po_qty * po_unit, 2),
                           "freight_terms": "freight prepaid by vendor, not separately billable"},
        "delivery": {"date": "2026-03-01", "received_qty": received_qty, "condition": condition},
        "payment": {"terms": "net 30", "status": status, "days_until_due": days_until_due,
                    "early_payment_discount_pct": discount_pct, "discount_expires_in_days": discount_days},
        "vendor_history": {"invoices_12m": 12, "disputes_12m": disputes, "prior_invoice_ids": list(prior_ids)},
    }

examples = {
    "clean match":            make_state(),
    "duplicate invoice":      make_state(inv_id="INV-2026-4242", prior_ids=["INV-2026-4242", "INV-2026-3100"]),
    "over-billed price":      make_state(inv_unit=65.0),
    "short delivery":         make_state(received_qty=6),
    "overdue, discount soon": make_state(days_until_due=-5, status="overdue", discount_pct=2, discount_days=1),
}
for name, st in tqdm(examples.items(), desc="examples", unit="case"):
    print(f"\\n=== {name}")
    show(decide(st, QUESTIONS))
""")

md("""
### Your input
Edit the dict below (any fields in the dataset schema) and re-run the cell.
You can also pass a subset of questions, or write your own question in the same typed format.
""")
code("""
my_state = {
    "invoice": {"id": "INV-2026-7777", "vendor": "Northwind Traders", "currency": "USD",
                "lines": [{"sku": "SKU-210", "qty": 20, "unit_usd": 12.5, "total_usd": 250.0},
                          {"sku": "FREIGHT", "qty": 1, "unit_usd": 45.0, "total_usd": 45.0}],
                "total_usd": 295.0},
    "purchase_order": {"id": "PO-3141", "lines": [{"qty": 20, "unit_usd": 12.5}], "total_usd": 250.0,
                       "freight_terms": "freight prepaid by vendor, not separately billable"},
    "delivery": {"date": "2026-10-01", "received_qty": 20, "condition": "accepted"},
    "payment": {"terms": "net 30", "status": "unpaid", "days_until_due": 3,
                "early_payment_discount_pct": None, "discount_expires_in_days": None},
    "vendor_history": {"invoices_12m": 30, "disputes_12m": 1, "prior_invoice_ids": ["INV-2026-7001"]},
}

show(decide(my_state, QUESTIONS))
""")
code("""
# Or paste a case as JSON text (e.g. a row's `state` copied from the dataset) ...
my_json = \"\"\"
PASTE_CASE_JSON_HERE
\"\"\"
my_json_questions = QUESTIONS   # for another workflow, paste that row's `questions` here as a dict
if "PASTE" not in my_json:
    show(decide(json.loads(my_json), my_json_questions))

# ... and/or ask a custom question (types: "choice" with a criteria dict, "noul" true/false, "score" with a list)
my_questions = {
    "needs_vendor_contact": {"type": "noul",
                             "instructions": "Someone should contact the vendor before this invoice is paid.",
                             "criteria": {"true": "The vendor must clarify or correct something.",
                                          "false": "No vendor contact is needed."}},
}
show(decide(my_state, my_questions))
""")

md("""
## 9. Publish to the Hugging Face Hub (leaderboard)
The benchmark is a Hub **Benchmark** dataset (it ships an `eval.yaml` with tasks `accuracy`, `kl_from_gold`, `brier`
and `ece` on `all/test`). A model repo that contains `.eval_results/*.yaml` with those task ids is picked up by the
dataset's leaderboard automatically. The README tables are curated by hand, so the dataset card also asks you to
open a discussion with the numbers and the mode; the last cell does that.

This model was fine-tuned on `train`, so it belongs to the **"Fitted or fine-tuned on `train`"** table.
Format follows the [Hub eval-results docs](https://huggingface.co/docs/hub/en/eval-results): one entry per task id,
values as fractions (as other entries on this leaderboard do), the dataset revision we scored against, and a
`source` link to the full test predictions in the model repo.
Nothing is uploaded until you flip the flags below. Needs a token with write access (`uv run hf auth login`).
""")
code("""
import os
from datetime import date
from huggingface_hub import HfApi, whoami

BENCHMARK       = "LocalLLaMA/typed-decisions"
HF_REPO         = f"{whoami()['name']}/{os.path.basename(OUT_DIR)}"
PRIVATE         = False   # the leaderboard only lists public models
PUSH_TO_HUB     = False   # set True to create/update the model repo with weights, model card and eval results
OPEN_DISCUSSION = False   # set True to post the scores on the benchmark's discussion board

api = HfApi()
DATASET_SHA = api.dataset_info(BENCHMARK).sha
overall = fine_tuned.loc["OVERALL"].copy()
# The README's reference rows (Uniform 0.169, Prior 0.088) are close to ECE computed per question and averaged
# over the 20 questions, not pooled over all 2,000 decisions; report that one to the leaderboard, keep both.
overall["ece_pooled"] = overall.ece
overall["ece"] = fine_tuned.drop("OVERALL").ece.mean()
by_type, by_wf = score(test_ex, ft_preds, by="qtype"), score(test_ex, ft_preds, by="workflow")
print(HF_REPO, "| dataset revision", DATASET_SHA[:8])
overall
""")

md("Write the predictions, metrics, `.eval_results/` file and model card into `OUT_DIR` (next to the saved weights).")
code("""
import yaml

# full test distributions, one line per case, so anyone can re-score
with open(f"{OUT_DIR}/test_predictions.jsonl", "w") as f:
    for cid, items in tqdm(by_case.items(), desc="write predictions", unit="case"):
        f.write(json.dumps({"id": cid, "workflow": items[0][0]["workflow"], "answers": {
            e["qname"]: {"label": e["labels"][int(p.argmax())],
                         "probabilities": dict(zip(e["labels"], map(float, p)))} for e, p in items}}) + "\\n")

metrics = {"config": CONFIG, "split": "test", "n_cases": len(test_raw), "n_decisions": len(test_ex),
           "overall": overall.drop("n").to_dict(), "ece_definition": "10-bin max-probability ECE per question, averaged over questions; ece_pooled = over all decisions", "by_type": by_type.drop(columns="n").to_dict("index"),
           "by_workflow": by_wf.drop(columns="n").to_dict("index"), "ms_per_case": ms_per_case, "device": DEVICE}
json.dump(metrics, open(f"{OUT_DIR}/test_metrics.json", "w"), indent=1)

# Hub eval results -> benchmark leaderboard
os.makedirs(f"{OUT_DIR}/.eval_results", exist_ok=True)
NOTES = (f"Fine-tuned on the Typed Decisions train split ({MODEL_ID}, full fine-tune) - NOT zero-shot; belongs to the "
         f"'fitted or fine-tuned on train' table. Test split, {len(test_raw)} cases / {len(test_ex)} decisions. "
         "One forward pass per question (not all five in one request); distribution = softmax over the option-letter "
         "logits. ECE = 10-bin max-probability ECE per question, averaged over the 20 questions "
         f"(pooled over all decisions: {overall.ece_pooled:.4f}). Self-reported, no verifyToken.")
SOURCE = {"url": f"https://huggingface.co/{HF_REPO}/blob/main/test_predictions.jsonl",
          "name": "Full test distributions", "user": HF_REPO.split("/")[0]}
eval_results = [{"dataset": {"id": BENCHMARK, "task_id": task, "revision": DATASET_SHA},
                 "value": round(float(overall[col]), 4), "date": date.today().isoformat(),
                 "source": dict(SOURCE), "notes": NOTES}   # copies: no YAML anchors/aliases
                for task, col in [("accuracy", "accuracy"), ("kl_from_gold", "kl"), ("brier", "brier"), ("ece", "ece")]]
yaml.safe_dump(eval_results, open(f"{OUT_DIR}/.eval_results/typed-decisions.yaml", "w"), sort_keys=False)

def md_table(df):
    df = df.drop(columns="n", errors="ignore")
    head = "| | " + " | ".join(df.columns) + " |\\n|---|" + "---|" * len(df.columns) + "\\n"
    return head + "\\n".join(f"| {i} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for i, r in df.iterrows())

RESULTS_MD = (f"Scored on `{BENCHMARK}` `all/test` ({len(test_raw)} cases, {len(test_ex)} decisions), "
              f"never seen in training.\\n\\n"
              f"| Model | Kind | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | Latency |\\n|---|---|---|---|---|---|---|\\n"
              f"| [{HF_REPO.split('/')[-1]}](https://huggingface.co/{HF_REPO}) (Gemma 3 270M, full fine-tune) | "
              f"fine-tuned on `train` | {overall.accuracy:.3f} | {overall.kl:.4f} | {overall.brier:.4f} | {overall.ece:.4f} | "
              f"{ms_per_case:.0f} ms / case ({DEVICE}, Apple Silicon) |\\n\\n"
              f"ECE is computed per question and averaged over the 20 questions (closest to the README's reference rows); "
              f"pooled over all decisions it is {overall.ece_pooled:.4f}.\\n\\n"
              f"**By question type**\\n\\n{md_table(by_type)}\\n\\n**By workflow**\\n\\n{md_table(by_wf)}\\n\\n"
              f"Baselines on the same split: Prior accuracy {comparison.loc['Prior', 'accuracy']:.3f} / "
              f"KL {comparison.loc['Prior', 'kl']:.3f}; zero-shot Gemma 3 270M accuracy "
              f"{comparison.loc['Gemma-270M zero-shot', 'accuracy']:.3f} / KL {comparison.loc['Gemma-270M zero-shot', 'kl']:.3f}.")

CARD = f\"\"\"---
base_model: {MODEL_ID}
datasets:
- {BENCHMARK}
license: gemma
library_name: transformers
tags:
- typed-decisions
- calibration
- structured-decisions
- probabilistic-classification
---

# {HF_REPO.split('/')[-1]}

[Gemma 3 270M]({"https://huggingface.co/" + MODEL_ID}) fully fine-tuned as a **probabilistic decision model** for
[{BENCHMARK}](https://huggingface.co/datasets/{BENCHMARK}). Mode: **fine-tuned on `train`** (all four workflows,
{len(train_ex) // 5} cases for training and {len(val_ex) // 5} held out for epoch selection on validation KL).

## Results

{RESULTS_MD}

Full test distributions: `test_predictions.jsonl`; metrics: `test_metrics.json`.
Gold is the mean of three teacher samples, so scores measure agreement with that teacher (self-agreement ≈ 0.735).

## How it works

Each (case, question) pair is one prompt: the case JSON, the question's instructions, and its options listed as
letters `A`, `B`, `C`, … with their descriptions. The model's next-token logits at the reply position, restricted to
those letter tokens and softmaxed, are the answer distribution. Training minimises soft cross-entropy against the
gold distribution. One forward pass per question (5 per case).

## Usage

```python
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "{HF_REPO}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation="eager").eval()
LETTERS = "ABCDEFGHIJ"
SYSTEM = {SYSTEM!r}

def options(q):
    if q["type"] == "choice": return list(q["criteria"].items())
    if q["type"] == "noul":
        c = q.get("criteria") or {{}}
        return [("true", c.get("true", "The statement is true.")), ("false", c.get("false", "The statement is false."))]
    return [(str(i), d) for i, d in enumerate(q["criteria"])]

@torch.no_grad()
def decide(state, questions):
    out = {{}}
    for name, q in questions.items():
        opts = options(q)
        kind = {{"choice": "Choose one option.", "noul": "Is the statement true or false?",
                "score": "Pick the level on the rubric."}}[q["type"]]
        prompt = (f"{{SYSTEM}}\\\\n\\\\n### Case\\\\n{{json.dumps(state, separators=(',', ':'))}}\\\\n\\\\n"
                  f"### Question ({{name}})\\\\n{{q['instructions']}}\\\\n{{kind}}\\\\n\\\\n### Options\\\\n"
                  + "\\\\n".join(f"{{LETTERS[i]}}. {{l}}: {{d}}" for i, (l, d) in enumerate(opts))
                  + "\\\\n\\\\nAnswer with one letter.")
        text = tok.apply_chat_template([{{"role": "user", "content": prompt}}], tokenize=False, add_generation_prompt=True)
        logits = model(**tok(text, add_special_tokens=False, return_tensors="pt")).logits[0, -1]
        ids = [tok.encode(LETTERS[i], add_special_tokens=False)[0] for i in range(len(opts))]
        p = torch.softmax(logits[ids].float(), -1).tolist()
        out[name] = dict(zip([l for l, _ in opts], p))
    return out
```
\"\"\"
open(f"{OUT_DIR}/README.md", "w").write(CARD)
print(open(f"{OUT_DIR}/.eval_results/typed-decisions.yaml").read())
print(RESULTS_MD)
""")

code("""
if PUSH_TO_HUB:
    api.create_repo(HF_REPO, private=PRIVATE, exist_ok=True)
    api.upload_folder(folder_path=OUT_DIR, repo_id=HF_REPO,
                      commit_message=f"Gemma 3 270M typed-decisions: acc {overall.accuracy:.3f}, KL {overall.kl:.4f}")
    print("pushed: https://huggingface.co/" + HF_REPO)
    # the benchmark's leaderboard API should list the model (per task) shortly after the push
    import requests
    for task in ("accuracy", "kl_from_gold"):
        lb = requests.get(f"https://huggingface.co/api/datasets/{BENCHMARK}/leaderboard", params={"task_id": task}).json()
        hit = [r for r in lb if r.get("modelId") == HF_REPO]
        print(task, "->", f"rank {hit[0]['rank']}, value {hit[0]['value']}" if hit else "not listed yet (can take a few minutes)")
else:
    print("PUSH_TO_HUB is False; nothing uploaded")
""")

code("""
DISCUSSION = f\"\"\"**Model:** https://huggingface.co/{HF_REPO}
**Mode:** fine-tuned on `train` (all four workflows), for the "Fitted or fine-tuned on `train`" table.

{RESULTS_MD}

**Setup.** Full fine-tune of `{MODEL_ID}` (fp32, AdamW, lr {LR}, {EPOCHS} epochs, epoch picked on held-out-train KL).
Each question is its own forward pass (5 per case, not all five in one request); the answer distribution is the softmax
of the next-token logits over the option letters. ECE is 10-bin expected calibration error of max probability vs.
correctness. Full test distributions are in `test_predictions.jsonl` in the model repo, so the numbers can be re-run.
\"\"\"
print(DISCUSSION)

if OPEN_DISCUSSION:
    d = api.create_discussion(BENCHMARK, repo_type="dataset",
                              title=f"Submission: {HF_REPO} (Gemma 3 270M, fine-tuned on train)", description=DISCUSSION)
    print("opened:", d.url)
""")

md("""
## Notes
* **Reading the score.** Gold is the average of three teacher samples; the dataset card puts teacher self-agreement at
  ~0.735 accuracy, so scores far above that mean the model is fitting teacher quirks.
* The leaderboard sends all five questions in one request; this notebook asks one question per forward pass
  (5 passes per case), which is the natural shape for a classifier-style fine-tune. Say so when submitting.
* 1,080 training cases are used (120 held out for epoch selection). Once hyperparameters are settled you can set
  `VAL_FRAC = 0` with a fixed `EPOCHS` to train on all 1,200 (the best-epoch logic then needs a val set, so keep
  the last epoch instead).
* To trade calibration for accuracy, select epochs on `val_acc` instead of `val_kl`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, "gemma3_270m_invoice_decisions.ipynb")
print("wrote gemma3_270m_invoice_decisions.ipynb")

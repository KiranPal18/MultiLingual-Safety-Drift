"""Train/test leak check: exact match + word 8-gram overlap between every training text and every eval prompt.
  .venv/bin/python scripts/11_leak_check.py   -> artifacts/logs/leak_check.md
"""
import json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D, N = ROOT / "artifacts/datasets", 8


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def norm(s):
    # drop punctuation/symbols only; a [^\w] regex would also strip Devanagari/Arabic vowel marks
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s.lower())
    return re.sub(r"\s+", " ", s).strip()


def grams(s):
    w = norm(s).split()
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


train = {
    "med_sft train": [m["content"] for r in jl(D / "med_sft/train/sft_messages.jsonl") for m in r["messages"]],
    "safety-mix refusals": [m["content"] for r in jl(D / "safety_refusals_curated.jsonl") for m in r["messages"]],
}
evals = {
    "medical MCQ": [r["question"] for r in jl(D / "eval_med_mcq.jsonl")],
    "general MCQ": [r["question"] for r in jl(D / "eval_general_mcq.jsonl")],
    "benign medical": [r["prompt"] for r in jl(D / "eval_benign_medical.jsonl")],
    "harmful (aya_redteaming)": [r["prompt"] for r in jl(D / "eval_harmful.jsonl")],
    "XSTest": [r["prompt"] for r in jl(D / "eval_xstest.jsonl")],
}
pool_ids = {r["id"] for r in jl(D / "safety_pool_sample.jsonl")}
eval_ids = {r["id"] for r in jl(D / "eval_harmful.jsonl")}
assert not pool_ids & eval_ids, "safety-mix prompts overlap the harmful eval set"

lines = ["## Train/test leak check", "",
         f"Exact match (after lowercasing and stripping punctuation) and shared word {N}-grams between each training "
         "set and each eval set. Safety-mix prompt IDs vs harmful-eval IDs: 0 shared.", "",
         "| train set | eval set | eval items | exact matches | items sharing a %d-gram |" % N, "|---|---|---|---|---|"]
for tn, tt in train.items():
    exact, tg = {norm(t) for t in tt}, set().union(*(grams(t) for t in tt))
    for en, et in evals.items():
        ex = sum(norm(e) in exact for e in et)
        ov = sum(bool(grams(e) & tg) for e in et)
        lines.append(f"| {tn} | {en} | {len(et)} | {ex} | {ov} ({100 * ov / len(et):.1f}%) |")
(ROOT / "artifacts/logs/leak_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

"""Generate responses + MCQ predictions for one model. All models are loaded the same way
(4-bit NF4, bf16 compute) so comparisons are fair.

  .venv/bin/python scripts/03_generate.py --name base
  .venv/bin/python scripts/03_generate.py --name sft --adapter artifacts/models/aya-med-sft/final
  .venv/bin/python scripts/03_generate.py --name resta --model artifacts/models/aya-med-resta
Writes artifacts/logs/eval/<name>/*.jsonl
"""
import argparse, json, os, random, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "artifacts/datasets"
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--model", default="CohereLabs/tiny-aya-global")
p.add_argument("--adapter", default=None)
p.add_argument("--bs", type=int, default=int(os.environ.get("GEN_BS", 16)))
p.add_argument("--max-new", type=int, default=256)
p.add_argument("--n-benign", type=int, default=100, help="benign medical prompts per language")
p.add_argument("--limit", type=int, default=None, help="debug: cap every set")
p.add_argument("--only", default=None, help="comma list of sets to run")
a = p.parse_args()
OUT = ROOT / "artifacts/logs/eval" / a.name
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")  # same tokenizer/chat template for every model
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    a.model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16))
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()


def load(name):
    rows = [json.loads(l) for l in open(D / name)]
    return rows[: a.limit] if a.limit else rows


def oom_safe(step, b):
    """Run step(batch); on CUDA OOM split the batch in half (LoRA adapters need more memory than the base)."""
    try:
        return step(b)
    except torch.OutOfMemoryError:
        if len(b) == 1:
            raise
    # retry outside the except block: the live exception would keep the failed batch's GPU tensors alive
    torch.cuda.empty_cache()
    h = len(b) // 2
    return oom_safe(step, b[:h]) + oom_safe(step, b[h:])


def chat(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen_batch(b):
    enc = tok([chat(r["prompt"]) for r in b], return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                         eos_token_id=STOP_IDS)
    return [{**r, "response": tok.decode(g, skip_special_tokens=True).strip()}
            for r, g in zip(b, gen[:, enc.input_ids.shape[1]:])]


@torch.no_grad()
def generate(rows, fname):
    out, t0 = [], time.time()
    rows = sorted(rows, key=lambda r: len(r["prompt"]))  # length-sorted batches = less padding
    for i in range(0, len(rows), a.bs):
        out += oom_safe(gen_batch, rows[i:i + a.bs])
        print(f"  {fname}: {len(out)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    with open(OUT / fname, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)


# generation_config only stops on <EOS_TOKEN>; fine-tuned models end turns with END_RESPONSE/END_OF_TURN
# (the chat template's assistant ending) and never emit <EOS>, so without these they loop into new turns.
STOP_IDS = tok.convert_tokens_to_ids(["<EOS_TOKEN>", "<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"])
assert tok.unk_token_id not in STOP_IDS, STOP_IDS
LETTERS = ["A", "B", "C", "D"]
LETTER_IDS = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]


@torch.no_grad()
def mcq(rows, fname):
    """Zero-shot: pick the letter with the highest next-token logit after the assistant turn opens."""
    def step(b):
        prompts = [chat(r["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, r["choices"]))
                        + "\nAnswer with only the letter (A, B, C or D).") for r in b]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, LETTER_IDS].float()
        return [{**r, "pred": int(lg.argmax()), "correct": int(lg.argmax()) == r["target"]} for r, lg in zip(b, logits)]

    out = []
    for i in range(0, len(rows), a.bs):
        out += oom_safe(step, rows[i:i + a.bs])
    acc = sum(r["correct"] for r in out) / len(out)
    print(f"  {fname}: acc={acc:.3f} (n={len(out)})")
    with open(OUT / fname, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)


rng = random.Random(0)
benign = load("eval_benign_medical.jsonl")
by_lang = {}
for r in benign:
    by_lang.setdefault(r["lang"], []).append(r)
benign = [r for l in by_lang.values() for r in rng.sample(l, min(a.n_benign, len(l)))]

jobs = {
    "med_mcq": lambda: mcq(load("eval_med_mcq.jsonl"), "med_mcq.jsonl"),
    "general_mcq": lambda: mcq(load("eval_general_mcq.jsonl"), "general_mcq.jsonl"),
    "harmful": lambda: generate(load("eval_harmful.jsonl"), "harmful.jsonl"),
    "xstest": lambda: generate(load("eval_xstest.jsonl"), "xstest.jsonl"),
    "benign_medical": lambda: generate(benign, "benign_medical.jsonl"),
}
# Not part of the default eval: answers to held-out red-team prompts, used to build safety-mix training data.
EXTRA = {"safety_pool": lambda: generate(load("safety_pool_sample.jsonl"), "safety_pool.jsonl")}
selected = a.only.split(",") if a.only else list(jobs)
jobs.update(EXTRA)
for k in selected:
    print(f"[{a.name}] {k}")
    jobs[k]()
json.dump(vars(a), open(OUT / "run_args.json", "w"), indent=2)

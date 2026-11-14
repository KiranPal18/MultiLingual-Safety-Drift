"""Prepare (NOT upload) Hugging Face repos: model cards in the model folders, dataset folders under artifacts/hf/.
  .venv/bin/python scripts/10_prepare_hf.py --user <your-hf-username>
Uploading is done by hand - see UPLOAD_CHECKLIST.md.
"""
import argparse, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A, D, L, M = ROOT / "artifacts", ROOT / "artifacts/datasets", ROOT / "artifacts/logs", ROOT / "artifacts/models"
p = argparse.ArgumentParser()
p.add_argument("--user", default="KiranPal18")
p.add_argument("--github", default="<GitHub repo link>", help="URL of the project's GitHub repo")
a = p.parse_args()
U = a.user
R = json.load(open(L / "results.json"))["overall"]
SFT_DS, EVAL_DS = f"{U}/aya-med-multilingual-sft", f"{U}/aya-med-safety-evals"


def pct(metric, model):
    v = R.get(metric, {}).get(model)
    return "-" if v is None else f"{100 * v:.1f}"


def table(model):
    rows = [("Medical accuracy (Global-MMLU, 5 langs) ↑", "med_acc"), ("General accuracy ↑", "general_acc"),
            ("Harmful ASR, aya_redteaming (Qwen3Guard) ↓", "harmful_ASR"),
            ("Harmful ASR (Llama-Guard-3-1B) ↓", "harmful_ASR_llamaguard"), ("Harmful refusal ↑", "harmful_refusal"),
            ("XSTest unsafe ASR ↓", "xstest_unsafe_ASR"), ("XSTest over-refusal ↓", "xstest_over_refusal"),
            ("Benign-medical over-refusal ↓", "benign_med_over_refusal")]
    out = ["| Metric (%) | Tiny Aya Global | This model |", "|---|---|---|"]
    out += [f"| {n} | {pct(k, 'base')} | {pct(k, model)} |" for n, k in rows]
    return "\n".join(out)


COMMON_FOOT = f"""
## How it was evaluated
Every model was evaluated the same way: loaded in 4-bit NF4, greedy decoding, Tiny Aya's own chat template, in
English, Hindi, Spanish, French and Arabic. Generation stops on `<EOS_TOKEN>`, `<|END_RESPONSE|>` and
`<|END_OF_TURN_TOKEN|>`. That last part matters: fine-tuned Tiny Aya models never emit `<EOS_TOKEN>`, so without the
other two stop tokens they keep going and loop. Answers were judged by Qwen3Guard-Gen-4B, which I checked against
Llama-Guard-3-1B and 96 responses I labelled by hand. All prompts, generations and labels are in
[{EVAL_DS}](https://huggingface.co/datasets/{EVAL_DS}).

## What it's for, and what it isn't
This is a research artifact for studying how a domain fine-tune changes safety behaviour across languages.
**Please don't use it for medical advice.** The results come from a single seed and 80 red-team prompts per
language, and both judges miss some unsafe answers, so read the attack success rates as lower bounds.
Licence: CC-BY-NC-4.0, inherited from Tiny Aya. Full write-up and code: {a.github}
"""


def adapter_card(name, model_key, title, summary, data_line):
    return f"""---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language: [en, hi, es, fr, ar]
pipeline_tag: text-generation
tags: [lora, qlora, sft, medical, multilingual, ai-safety, aligntune, lexsi-hackathon]
datasets: [{SFT_DS}]
---

# {title}

{summary}

This is a QLoRA adapter for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), trained
with AlignTune. It's part of *"Fine-tuning breaks safety, in every language"*, my project for the Lexsi AI Safety &
Alignment Hackathon.

## Results

{table(model_key)}

## Training
- Data: {data_line}
- QLoRA NF4 + double quant, bf16 compute; LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down projections
- lr 2e-4, 1 epoch, effective batch 16, max length 640, paged AdamW 8-bit, gradient checkpointing, seed 42
- The tied 262k-token embedding is kept in bf16. PEFT would otherwise up-cast it to fp32, which runs out of memory on 8 GB.
- Trained on a laptop RTX 4060 (8 GB)

## Usage
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{U}/{name}")
stop = tok.convert_tokens_to_ids(["<EOS_TOKEN>", "<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"])
msgs = [{{"role": "user", "content": "What are common symptoms of iron-deficiency anaemia?"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False, eos_token_id=stop)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```
{COMMON_FOOT}"""


cards = {
    M / "aya-med-sft/final": adapter_card(
        "tiny-aya-global-med-sft", "sft", "Tiny Aya Global — medical SFT (safety-degraded, for research)",
        "> ⚠️ **I'm publishing this model as evidence, not for use.** It was fine-tuned only on harmless medical Q&A, and "
        "that alone nearly doubled its attack success rate on red-team prompts (10.5% → 18.5%). It also refuses less "
        "in all five languages. If you want the medical model, use "
        f"[{U}/tiny-aya-global-med-safemix](https://huggingface.co/{U}/tiny-aya-global-med-safemix) instead, which "
        "fixes this.",
        "4,497 multilingual medical Q&A pairs from ApolloCorpus, curated with CuratorKIT (`med_sft` config)"),
    M / "aya-med-sft-safemix/final": adapter_card(
        "tiny-aya-global-med-safemix", "safemix", "Tiny Aya Global — safety-mixed medical SFT (recommended)",
        "Plain medical fine-tuning made Tiny Aya less safe in every language I tested. This model uses the same recipe "
        "with one change: 309 refusals in five languages, about 6% of the data, are mixed into the training set. The "
        "original model wrote those refusals itself. The result is an attack success rate of 4.0%, lower than the "
        "Tiny Aya Global I started from (10.5%), with medical accuracy unchanged.",
        "4,497 medical Q&A pairs plus 309 refusals. Tiny Aya Global wrote the refusals on held-out aya_redteaming "
        "prompts, and I kept only the ones Qwen3Guard judged both safe and a refusal. Curated with CuratorKIT "
        "(`safety_mix` config)."),
}
cards[M / "aya-med-resta"] = f"""---
base_model: CohereLabs/tiny-aya-global
license: cc-by-nc-4.0
language: [en, hi, es, fr, ar]
pipeline_tag: text-generation
tags: [resta, safetune, model-merging, medical, multilingual, ai-safety, lexsi-hackathon]
---

# Tiny Aya Global — medical SFT + RESTA (α = 0.25)

This model starts from the [medical SFT adapter](https://huggingface.co/{U}/tiny-aya-global-med-sft), merged into
Tiny Aya Global. I then tried to repair its safety in weight space with SafeTune's RESTA, adding back part of the
difference between the aligned and the base model: θ = θ_sft + 0.25 · (θ_tiny-aya-global − θ_tiny-aya-base).
These are full bf16 weights.

> This one is a negative result, and I'm publishing it for completeness. RESTA brings back some safety, but it
> costs about 8 accuracy points and the model refuses three times as many safe XSTest prompts. The repair is also
> uneven: in French it ends up less safe than the plain fine-tune. The
> [safety-mixed model](https://huggingface.co/{U}/tiny-aya-global-med-safemix) does better on every count.

## Results

{table("resta")}

At α = 0.1 (not uploaded): medical/general accuracy {pct("med_acc", "resta_a0.1")}/{pct("general_acc", "resta_a0.1")},
harmful ASR {pct("harmful_ASR", "resta_a0.1")}. Accuracy survives, but safety barely improves.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{U}/tiny-aya-global-med-resta")
model = AutoModelForCausalLM.from_pretrained("{U}/tiny-aya-global-med-resta", dtype="bfloat16", device_map="auto")
# generate with eos_token_id = ids of <EOS_TOKEN>, <|END_RESPONSE|>, <|END_OF_TURN_TOKEN|>
```
{COMMON_FOOT}"""
for d, text in cards.items():
    (d / "README.md").write_text(text, encoding="utf-8")
    print("card ->", (d / "README.md").relative_to(ROOT))

# ---- dataset 1: training data -------------------------------------------------------------------
H = A / "hf"
sft = H / "aya-med-multilingual-sft"
shutil.rmtree(sft, ignore_errors=True)
for split in ["train", "val"]:
    (sft / "med_sft").mkdir(parents=True, exist_ok=True)
    shutil.copy(D / f"med_sft/{split}/sft_messages.jsonl", sft / f"med_sft/{split}.jsonl")
(sft / "safety_mix").mkdir()
shutil.copy(D / "med_sft_safetymix.jsonl", sft / "safety_mix/train.jsonl")
shutil.copy(D / "safety_refusals_curated.jsonl", sft / "safety_mix/refusals_only.jsonl")
for f in ["manifest.json", "checksums.txt", "lexsi_provenance.json"]:
    shutil.copy(D / "med_sft" / f, sft / f"med_sft/curatorkit_{f}")
    shutil.copy(D / "med_sft_safetymix" / f, sft / f"safety_mix/curatorkit_{f}")
n = {k: sum(1 for _ in open(v, encoding="utf-8")) for k, v in
     {"tr": sft / "med_sft/train.jsonl", "va": sft / "med_sft/val.jsonl", "mix": sft / "safety_mix/train.jsonl",
      "ref": sft / "safety_mix/refusals_only.jsonl"}.items()}
(sft / "README.md").write_text(f"""---
license: cc-by-nc-4.0
language: [en, hi, es, fr, ar]
task_categories: [text-generation]
tags: [medical, multilingual, sft, safety, refusals, curatorkit, lexsi-hackathon]
configs:
- config_name: med_sft
  default: true
  data_files:
  - split: train
    path: med_sft/train.jsonl
  - split: validation
    path: med_sft/val.jsonl
- config_name: safety_mix
  data_files:
  - split: train
    path: safety_mix/train.jsonl
- config_name: refusals_only
  data_files:
  - split: train
    path: safety_mix/refusals_only.jsonl
---

# Multilingual medical SFT data + safety mix (Tiny Aya)

This is the training data behind [{U}/tiny-aya-global-med-sft](https://huggingface.co/{U}/tiny-aya-global-med-sft)
and [{U}/tiny-aya-global-med-safemix](https://huggingface.co/{U}/tiny-aya-global-med-safemix), in chat `messages`
format.

| Config | Rows | Content |
|---|---|---|
| `med_sft` | {n['tr']} train / {n['va']} val | medical Q&A in en, hi, es, fr, ar |
| `safety_mix` | {n['mix']} | `med_sft` train + refusals, curated again |
| `refusals_only` | {n['ref']} | the refusals alone (prompt + refusal) |

## How it was built (CuratorKIT)
- **Medical Q&A** from [FreedomIntelligence/ApolloCorpus](https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus)
  (Apache-2.0): en/es exam Q&A (1000 each), fr exam Q&A (759), ar patient Q&A (1000, length-filtered),
  hi medical wiki Q&A + general Hindi Q&A filtered by a medical-keyword regex (1000).
- CuratorKIT: schema gate, exact + MinHash dedup (threshold 0.85), text cleaning, 3–1024 token filter,
  95/5 split. 4759 → 4734 rows. CuratorKIT manifests, checksums and provenance are in each folder.
- **Refusals:** 500 held-out [CohereLabs/aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming)
  prompts (100 per language, none of them used in evaluation), answered by Tiny Aya Global itself. I kept only the
  replies Qwen3Guard-Gen-4B judged both *Safe* and a *Refusal*, which left {n['ref']}. I didn't write any harmful text
  myself.

## Notes
- `refusals_only` pairs red-team prompts from aya_redteaming with refusals. The responses don't contain harmful
  instructions, but the prompts are offensive on purpose.
- Language labels were not preserved through curation.
- Licence: CC-BY-NC-4.0 (contains Tiny Aya outputs); sources: ApolloCorpus and aya_redteaming (Apache-2.0).
""", encoding="utf-8")
print("dataset ->", sft.relative_to(ROOT))

# ---- dataset 2: evaluation prompts, generations, judge labels, human labels --------------------
ev = H / "aya-med-safety-evals"
shutil.rmtree(ev, ignore_errors=True)
(ev / "prompts").mkdir(parents=True)
for f in ["eval_med_mcq", "eval_general_mcq", "eval_harmful", "eval_xstest", "eval_benign_medical", "safety_pool_sample"]:
    shutil.copy(D / f"{f}.jsonl", ev / f"prompts/{f}.jsonl")
MODELS = ["base", "sft", "resta", "resta_a0.1", "safemix"]
for m in MODELS + ["aligned_safety_pool"]:
    (ev / f"generations/{m}").mkdir(parents=True)
    for f in (L / "eval" / m).glob("*.jsonl"):
        shutil.copy(f, ev / f"generations/{m}/{f.name}")
shutil.copytree(L / "judge_check", ev / "judge_check")
(ev / "judge_check/human_labels.csv").unlink(missing_ok=True)  # unlabelled template; the labelled file is kept
for f in ["results.md", "results.json"]:
    shutil.copy(L / f, ev / f)
gen_cfg = "\n".join(f"""- config_name: {m.replace('.', '')}_{s}
  data_files: generations/{m}/{s}.jsonl""" for m in MODELS for s in ["harmful", "xstest", "benign_medical"])
(ev / "README.md").write_text(f"""---
license: cc-by-nc-4.0
language: [en, hi, es, fr, ar]
task_categories: [text-generation]
tags: [ai-safety, red-teaming, llm-as-judge, multilingual, medical, xstest, lexsi-hackathon]
configs:
- config_name: prompts_harmful
  default: true
  data_files: prompts/eval_harmful.jsonl
- config_name: prompts_xstest
  data_files: prompts/eval_xstest.jsonl
- config_name: prompts_benign_medical
  data_files: prompts/eval_benign_medical.jsonl
- config_name: prompts_med_mcq
  data_files: prompts/eval_med_mcq.jsonl
- config_name: prompts_general_mcq
  data_files: prompts/eval_general_mcq.jsonl
{gen_cfg}
---

# Multilingual medical safety-drift evaluations (Tiny Aya)

> ⚠️ This contains red-team prompts and some **unsafe model outputs**, since that's what is being measured.
> Please use it for safety research only.

This is everything you need to check the results of *"Fine-tuning breaks safety, in every language"*:
- the evaluation prompts
- every model's answers, with the judge labels
- my blind hand labels
- the judge-agreement analysis

| Folder | Content |
|---|---|
| `prompts/` | Global-MMLU medical (200×5 langs) and general (150×5) MCQ, aya_redteaming harmful (80×5), XSTest (450, en), benign medical questions, held-out safety-mix prompts |
| `generations/<model>/` | responses (`response`), MCQ predictions (`pred`, `correct`), Qwen3Guard labels (`guard_safety`, `guard_refusal`, `guard_categories`, `guard_raw`) and Llama-Guard-3-1B labels (`lg_safety`) |
| `judge_check/` | `human_labels_final.csv` (96 blind human labels, en+hi), `agreement.md/json` |
| `results.md/json` | all metrics, per-language slices, paired-bootstrap p-values (AuditKIT) |

Models: `base` = CohereLabs/tiny-aya-global, `sft` = [{U}/tiny-aya-global-med-sft](https://huggingface.co/{U}/tiny-aya-global-med-sft),
`safemix` = [{U}/tiny-aya-global-med-safemix](https://huggingface.co/{U}/tiny-aya-global-med-safemix),
`resta` = [{U}/tiny-aya-global-med-resta](https://huggingface.co/{U}/tiny-aya-global-med-resta) (α=0.25), `resta_a0.1` = RESTA α=0.1.
Every answer was generated in 4-bit NF4 with greedy decoding and at most 256 new tokens.

Sources: Global-MMLU, aya_redteaming (Apache-2.0), XSTest (CC-BY-4.0). Model outputs: CC-BY-NC-4.0 (Tiny Aya).
""", encoding="utf-8")
print("dataset ->", ev.relative_to(ROOT))

# ---- privacy: strip the local project path from provenance/manifests in everything that gets uploaded --------
for d in [*cards, H]:
    for f in d.rglob("*"):
        if f.suffix in {".json", ".yaml", ".txt", ".md"} and f.stat().st_size < 5_000_000:
            t = f.read_text(encoding="utf-8")
            if str(ROOT) in t:
                f.write_text(t.replace(str(ROOT) + "/", "").replace(str(ROOT), "."), encoding="utf-8")
                print("scrubbed local path ->", f.relative_to(ROOT))

# Fine-tuning breaks safety, in every language
### A multilingual healthcare case study on Tiny Aya — measure the drift, then repair it

**Lexsi AI Safety & Alignment Hackathon — Track 1 (Fine-tuning & Data Curation) + Track 2 (Red Teaming, Safety & Alignment)**

Kits chained: **CuratorKIT → AlignTune → AuditKIT → SafeTune** (4 kits). Model family: **Cohere Tiny Aya** only.
Languages: **English, Hindi, Spanish, French, Arabic**.

| | |
|---|---|
| Models | https://huggingface.co/collections/KiranPal18/tiny-aya-multilingual-safety-drift-models-6af8e4969994d1dfbb96e621 |
| Datasets | https://huggingface.co/collections/KiranPal18/tiny-aya-multilingual-safety-drift-datasets-6af8e4a1c9b6f21539beae5a |
| Code | this repository |

![Harmful-prompt ASR per language](docs/asr_by_language.png)

---

## Abstract

Fine-tuning an aligned model on benign, domain-specific data is the most common way small models are deployed
in practice — and it is known to erode safety alignment. Almost all evidence for this is in English. We ask
whether the erosion happens **across languages**, in a setting where it matters: a **multilingual medical
assistant** built from Cohere's 3.35B **Tiny Aya Global**.

We QLoRA fine-tune Tiny Aya Global on 4.5k curated medical Q&A pairs in five languages (CuratorKIT + AlignTune)
and evaluate it before and after on medical and general knowledge (Global-MMLU), multilingual red-team prompts
(aya_redteaming), XSTest, and benign medical questions, with a multilingual safety judge (Qwen3Guard-Gen-4B) and
paired-bootstrap significance tests (AuditKIT). Everything is evaluated in the **4-bit NF4 deployment
configuration**.

**Result 1 — drift is real and multilingual.** The medical fine-tune keeps medical accuracy (52.0 → 52.8%) but
the harmful-prompt **attack success rate (ASR) nearly doubles, 10.5% → 18.5% (p = 0.001)**, rising in all five
languages (largest in English +12.5 pts, Hindi +12.5, Arabic +10), refusals drop 58.5% → 45.8%, and unsafe
compliance on XSTest's unsafe prompts rises 1% → 10%. No safety data was removed or attacked — the training data
was purely medical.

**Result 2 — data-level repair beats weight-level repair.** We compare two repairs from SafeTune's toolbox:
(a) **RESTA** weight-space re-alignment (θ_ft + α(θ_aligned − θ_base)) and (b) a **safety-mixed SFT** that adds
only 309 multilingual refusals (6% of the data), written by the original aligned model on held-out red-team prompts
and filtered by the judge. RESTA exposes a hard trade-off: α = 0.1 keeps accuracy but barely repairs safety
(ASR 17.8%), α = 0.25 partially repairs it (13.2%) but costs 7–10 accuracy points and triples XSTest
over-refusal (6.8 → 17.6%). The **safety-mixed SFT reaches ASR 4.0% — below the original model** — with refusal
up to 71.5%, XSTest unsafe 0%, no significant change in over-refusal on XSTest, and medical accuracy kept (52.8%),
at a cost of −3.7 pts general accuracy.

**Result 3 — the judge matters, so we checked it.** A second judge (Llama-Guard-3-1B) and 96 blind human labels
(en/hi) show Qwen3Guard is the better judge (87.5% agreement with the human vs 85.4%; refusal κ = 0.63). The human
labels reproduce the direction of the drift and of the repair. Unsafe-label κ is only ~0.3 (unsafe outputs are
rare, and both judges miss some), so absolute ASRs are likely **lower bounds**; relative comparisons are consistent
across judges for the repair, while the fine-tuning drift on aya_redteaming is visible under Qwen3Guard and the
human labels but not under Llama Guard.

We also document two engineering pitfalls that silently corrupt results on this model family (an fp32
up-cast of Tiny Aya's 262k-token tied embedding in PEFT, and a stop-token mismatch after fine-tuning that made
fine-tuned models loop) and how we fixed them.

---

## 1. Pipeline

```
ApolloCorpus (5 langs) ──► CuratorKIT ──► AlignTune QLoRA SFT ──► "Medical SFT"
   (minhash dedup, clean,       (Tiny Aya Global, r=16)         │
    length filter, export)                                      ├─► SafeTune RESTA α=0.25 / 0.1 (weight space)
                                                                │
aya_redteaming (held-out) ─► aligned model answers ─► Qwen3Guard keeps Safe+Refusal ─► CuratorKIT
                                                       ─► mixed with medical data ─► AlignTune SFT ─► "Safety-mixed SFT"

All models ──► generate (4-bit NF4, greedy) ──► Qwen3Guard + Llama Guard + human labels ──► AuditKIT report
```

| Kit | What it does here |
|---|---|
| **CuratorKIT** | MinHash dedup (0.85), cleaning, token-length filter, train/val split, `messages`/`alpaca` export with checksums and provenance — for both the medical set and the safety mix |
| **AlignTune** | QLoRA SFT (TRL backend), identical recipe for the plain and safety-mixed models |
| **AuditKIT** | Scores every metric on precomputed outputs, per-language slices, paired bootstrap significance vs the base model |
| **SafeTune** | `apply_resta` weight-space re-alignment using Tiny Aya Base → Tiny Aya Global as the safety vector |

## 2. Data

**Training (Track 1).** Medical Q&A from [ApolloCorpus](https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus)
(Apache-2.0): en/es exam Q&A (1000 each), fr exam Q&A (759), ar patient Q&A (1000, length-filtered), hi medical
wiki Q&A + medical-keyword-filtered general Hindi Q&A (1000). CuratorKIT: 4759 → 4734 kept → **4497 train / 237 val**.

**Safety mix.** 500 held-out [aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming) prompts
(100/language, disjoint from the eval prompts) answered by Tiny Aya Global; kept only replies Qwen3Guard labelled
*Safe* **and** *Refusal* → **309 refusals** (en 65, hi 54, es 61, fr 73, ar 56). Mixed with the medical train set
and curated again → 4804 examples. No harmful prompt or response was written by us.

**Evaluation.**

| Set | Source | Size | Measures |
|---|---|---|---|
| Medical MCQ | Global-MMLU (anatomy, clinical knowledge, college medicine, medical genetics, professional medicine, virology, nutrition) | 200 × 5 langs | medical accuracy (same question IDs in every language) |
| General MCQ | Global-MMLU, other subjects | 150 × 5 langs | general capability |
| Harmful | aya_redteaming | 80 × 5 langs | ASR, refusal rate |
| XSTest | walledai/XSTest | 250 safe + 200 unsafe (en) | unsafe compliance, over-refusal |
| Benign medical | Global-MMLU medical questions, no options | 100 × 5 langs | over-refusal on legitimate medical questions |

## 3. Setup

- **Model:** `CohereLabs/tiny-aya-global` (3.35B, Cohere2). Base for RESTA: `CohereLabs/tiny-aya-base`.
- **SFT:** QLoRA (NF4, double quant, bf16 compute), LoRA r=16, α=32, dropout 0.05 on all attention + MLP
  projections, lr 2e-4, 1 epoch, effective batch 16, max length 640, paged AdamW 8-bit, gradient checkpointing.
  Plain SFT: 282 steps, 51 min on an RTX 4060 8 GB (train loss 3.07 → 0.56).
- **RESTA:** LoRA merged into bf16 weights, then θ + α(θ_global − θ_base), α ∈ {0.25, 0.1}.
- **Evaluation:** every model loaded the same way — **4-bit NF4, bf16 compute** (the deployment configuration
  on consumer GPUs). Greedy decoding, ≤ 256 new tokens, the model's own chat template. MCQ: zero-shot, argmax over
  the next-token logits of A/B/C/D.
- **Judge:** `Qwen/Qwen3Guard-Gen-4B` (119 languages) on (prompt, response) → *Safe/Unsafe/Controversial* +
  *Refusal Yes/No*. **ASR** = share judged *Unsafe*. **Over-refusal** = share of benign prompts with *Refusal = Yes*.
- **Second judge:** `meta-llama/Llama-Guard-3-1B` (safe/unsafe only).
- **Significance:** AuditKIT paired bootstrap vs the base model (`p` shown in brackets).

## 4. Results

### Overall (%)

| Metric | Base (Tiny Aya Global) | Medical SFT | + RESTA α=0.25 | + RESTA α=0.1 | **Safety-mixed SFT** |
|---|---|---|---|---|---|
| Medical accuracy ↑ | 52.0 | 52.8 (p=.60) | 44.9 (p=.001) | 52.4 (p=.81) | **52.8** (p=.59) |
| General accuracy ↑ | 50.8 | 48.7 (p=.22) | 40.9 (p=.001) | 48.7 (p=.23) | 47.1 (p=.03) |
| Harmful ASR ↓ (Qwen3Guard) | 10.5 | **18.5** (p=.001) | 13.2 (p=.18) | 17.8 (p=.001) | **4.0** (p=.001) |
| Harmful ASR ↓ (Llama Guard) | 14.8 | 12.8 (p=.34) | 12.2 (p=.29) | 12.5 (p=.32) | **4.5** (p=.001) |
| Harmful refusal ↑ | 58.5 | 45.8 (p=.001) | 58.0 (p=.86) | 48.8 (p=.001) | **71.5** (p=.001) |
| XSTest unsafe ASR ↓ (Qwen3Guard) | 1.0 | **10.0** (p=.001) | 3.5 (p=.11) | 5.5 (p=.008) | **0.0** (p=.26) |
| XSTest unsafe ASR ↓ (Llama Guard) | 5.5 | 9.5 (p=.11) | 3.5 (p=.39) | 4.0 (p=.49) | 3.0 (p=.18) |
| XSTest over-refusal ↓ | 6.8 | 6.0 (p=.75) | **17.6** (p=.001) | 6.0 (p=.74) | 9.6 (p=.18) |
| Benign-medical over-refusal ↓ | 0.8 | 3.2 (p=.004) | 6.2 (p=.001) | 2.8 (p=.01) | 3.6 (p=.002) |

### Per language (%)

| Metric | Lang | Base | Medical SFT | RESTA 0.25 | RESTA 0.1 | Safety-mixed |
|---|---|---|---|---|---|---|
| Harmful ASR ↓ | en | 15.0 | 27.5 | 8.8 | 22.5 | 3.8 |
| | hi | 3.8 | 16.2 | 8.8 | 15.0 | 2.5 |
| | es | 16.2 | 20.0 | 15.0 | 20.0 | 7.5 |
| | fr | 10.0 | 11.2 | 17.5 | 12.5 | 0.0 |
| | ar | 7.5 | 17.5 | 16.2 | 18.8 | 6.2 |
| Harmful refusal ↑ | en | 48.8 | 32.5 | 60.0 | 41.2 | 67.5 |
| | hi | 65.0 | 47.5 | 66.2 | 57.5 | 82.5 |
| | es | 56.2 | 45.0 | 56.2 | 47.5 | 60.0 |
| | fr | 70.0 | 62.5 | 58.8 | 56.2 | 80.0 |
| | ar | 52.5 | 41.2 | 48.8 | 41.2 | 67.5 |
| Medical accuracy ↑ | en | 54.5 | 54.0 | 45.5 | 55.5 | 55.0 |
| | hi | 54.0 | 52.0 | 42.0 | 51.5 | 52.5 |
| | es | 52.0 | 55.5 | 47.0 | 57.0 | 54.0 |
| | fr | 51.0 | 54.5 | 49.0 | 53.0 | 54.0 |
| | ar | 48.5 | 48.0 | 41.0 | 45.0 | 48.5 |
| Benign-medical over-refusal ↓ | en | 1.0 | 2.0 | 5.0 | 1.0 | 1.0 |
| | hi | 0.0 | 5.0 | 4.0 | 6.0 | 6.0 |
| | es | 1.0 | 1.0 | 5.0 | 2.0 | 2.0 |
| | fr | 2.0 | 5.0 | 10.0 | 3.0 | 8.0 |
| | ar | 0.0 | 3.0 | 7.0 | 2.0 | 1.0 |

Full tables (incl. general accuracy and Llama Guard per language): [`artifacts/logs/results.md`](artifacts/logs/results.md).

### What the numbers say

1. **Benign medical fine-tuning erodes safety in all five languages.** ASR rises in every language (most in en,
   hi, ar; marginal in fr), refusal falls in every language, and XSTest unsafe compliance goes 1% → 10%. Hindi —
   the safest language before fine-tuning (3.8% ASR) — gets 4× worse.
2. **RESTA has no good operating point here.** α = 0.1 is almost a no-op on safety; α = 0.25 repairs part of it
   (unevenly: better in en/hi, *worse* than the SFT in French) while destroying ~8 accuracy points and tripling
   XSTest over-refusal. A single global safety vector does not transfer evenly across languages.
3. **A small multilingual safety mix fixes it.** 309 self-generated refusals (6% of the data) bring ASR below
   the original model in every language, under both judges and in the human sample, while keeping medical
   accuracy. Costs: −3.7 pts general accuracy and +2.8 pts benign-medical over-refusal (mostly hi/fr).

### Judge agreement check

| Comparison | n | Agreement | Cohen's κ |
|---|---|---|---|
| Qwen3Guard vs Llama Guard (unsafe), all responses | 4250 | 89.4% | 0.34 |
| **Qwen3Guard vs human** (unsafe) | 96 | **87.5%** | 0.33 |
| Llama Guard vs human (unsafe) | 96 | 85.4% | 0.28 |
| **Qwen3Guard vs human** (refusal) | 96 | 82.3% | **0.63** |

Human sample: 96 harmful-set responses (en + hi, 12 per model × language), shuffled, judge labels hidden.
Unsafe count among the 24 per model — human: base 1, SFT 4, RESTA 3, safety-mixed 2 (Qwen3Guard 3/4/3/0,
Llama Guard 5/3/3/1). Observations: Llama Guard flags 18.8% of base-model answers to XSTest's *safe* prompts as
unsafe (Qwen3Guard 1.2%), i.e. it over-flags benign answers; each judge misses 6 of the 10 human-labelled
unsafe responses, so reported ASRs are lower bounds. Per-language agreement:
[`artifacts/logs/judge_check/agreement.md`](artifacts/logs/judge_check/agreement.md).

### Results on the quantized model (T2b)

All numbers above are measured on the **4-bit NF4 quantized** models — the configuration that fits the 8 GB
consumer GPUs these small models are deployed on. We did not separately evaluate bf16, so we make no claim about
how much of the drift is caused by quantization itself.

## 5. Two pitfalls worth knowing (Tiny Aya / Cohere2)

1. **PEFT up-casts the tied 262k-token embedding to fp32.** `prepare_model_for_kbit_training` casts every
   non-quantized weight to fp32; on Tiny Aya that doubles the tied embedding/LM head to 2.1 GB and makes all logits
   fp32 → OOM on 8 GB even at batch 1. Keeping the frozen embedding in bf16 (a 5-line wrapper in
   `scripts/02_train_sft.py`) brings peak memory to 6.3 GB.
2. **Fine-tuned models never emit `<EOS_TOKEN>`.** Tiny Aya's `generation_config` stops only on `<EOS_TOKEN>`,
   but the chat template ends assistant turns with `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`, which is what SFT
   teaches. Fine-tuned models therefore start a new turn and repeat themselves until the token limit (222/500
   looping answers vs 1/500 for the base model) — which inflated our first ASR numbers. Fix: stop on all three
   tokens for every model (`STOP_IDS` in `scripts/03_generate.py`). The buggy run is kept in
   `artifacts/logs/eval_v1_no_stop_token/` for transparency.

## 6. Limitations

- 80 harmful prompts per language and one seed; per-language differences of a few points are within noise
  (overall comparisons carry bootstrap p-values).
- The safety mix was filtered with the same judge used for evaluation (Qwen3Guard). The independent judge
  (Llama Guard) and the human labels agree the safety-mixed model is safest, but some judge-specific bias cannot be
  ruled out. The safety-mix prompts are held out but from the same distribution as the harmful eval; XSTest is the
  out-of-distribution check.
- Human labels: one annotator (the author), English and Hindi only, 96 items.
- XSTest is English-only; benign-medical over-refusal relies on the judge's refusal label.
- The medical SFT did not improve medical MCQ accuracy (+0.8, n.s.): the training data is free-text Q&A, not
  multiple choice. The point of this project is the safety side-effect of a realistic fine-tune, not a stronger
  medical model. **None of these models are fit for medical use.**

## 7. Reproduce

```bash
# env (Python 3.12, CUDA GPU >= 8 GB)
uv venv -p 3.12 .venv && uv pip install -p .venv torch==2.13.0 -r scripts/remote/requirements.lock
for k in aligntune:b6e924e AuditKIT:556eb48 CuratorKIT:ed45ae3 SafeTune:cc3de5f; do   # kit commits we used
  git clone https://github.com/Lexsi-Labs/${k%%:*}.git third_party/${k%%:*} && git -C third_party/${k%%:*} checkout ${k##*:}; done
uv pip install -p .venv --no-deps -e third_party/aligntune -e third_party/AuditKIT -e third_party/CuratorKIT -e third_party/SafeTune
hf auth login          # accept the Tiny Aya, Qwen3Guard and (optional) Llama Guard licences first

.venv/bin/python scripts/01_build_data.py                 # data (CuratorKIT)
.venv/bin/python scripts/06_build_safety_mix.py sample    # held-out red-team prompts for the safety mix
bash scripts/run_all.sh                                   # SFT, eval, judge, safety mix, safety-mixed SFT, RESTA, report
bash scripts/resta_a01.sh                                 # optional RESTA α=0.1 ablation
.venv/bin/python scripts/08_judge_agreement.py llamaguard # second judge
.venv/bin/python scripts/08_judge_agreement.py sheet      # blind human-label sheet -> fill it in
.venv/bin/python scripts/08_judge_agreement.py score
.venv/bin/python scripts/09_plot.py
```
On a rented GPU: `scripts/remote.sh` (push / setup / login / run / status / pull).

| Script | Stage |
|---|---|
| `01_build_data.py` | download + CuratorKIT curation of training and eval sets |
| `02_train_sft.py` | AlignTune QLoRA SFT |
| `03_generate.py` | 4-bit generation + MCQ scoring |
| `04_judge.py` | Qwen3Guard judging |
| `05_repair_resta.py` | SafeTune RESTA |
| `06_build_safety_mix.py` | safety-mix sampling + curation |
| `07_report.py` | AuditKIT metrics, per-language slices, significance |
| `08_judge_agreement.py` | Llama Guard, human-label sheet, agreement |
| `09_plot.py` | figure |

## 8. Cost / ROI (optional)

Used a laptop GPU (RTX 4060, 8 GB): 4-bit QLoRA training and 4-bit evaluation fit in 8 GB of VRAM, and plain SFT
took 51 min. The safety-mixed repair costs one extra SFT run plus generating 500 answers — no
new human data and no extra model — and reduces harmful ASR by 78% relative to the plain fine-tune.

## Licences

Tiny Aya models and all derived weights/outputs: CC-BY-NC-4.0 (Cohere). ApolloCorpus, aya_redteaming,
Global-MMLU, Qwen3Guard: Apache-2.0. XSTest: CC-BY-4.0. Llama-Guard-3-1B: Llama 3.2 licence (used only as a judge).

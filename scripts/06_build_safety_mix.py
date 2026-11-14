"""Build the safety-mixed SFT set (medical data + ~10% multilingual refusals).

Step 1  sample held-out red-team prompts (disjoint from eval_harmful):
  .venv/bin/python scripts/06_build_safety_mix.py sample
Step 2  have the ALIGNED model answer them, then judge with Qwen3Guard:
  .venv/bin/python scripts/03_generate.py --name aligned_safety_pool --only safety_pool
  .venv/bin/python scripts/04_judge.py aligned_safety_pool
Step 3  keep only judged Safe + Refusal=Yes replies, mix with medical data, curate with CuratorKIT:
  .venv/bin/python scripts/06_build_safety_mix.py build
"""
import json, random, sys
from pathlib import Path

from curatorkit import Curator, CuratorConfig

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "artifacts/datasets"
PER_LANG = 100
rng = random.Random(7)


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"  {path.relative_to(ROOT)}: {len(rows)} rows")


if sys.argv[1] == "sample":
    pool = jl(D / "safety_pool_prompts.jsonl")
    out = []
    for lang in ["en", "hi", "es", "fr", "ar"]:
        rows = [r for r in pool if r["lang"] == lang]
        out += rng.sample(rows, PER_LANG)
    write(D / "safety_pool_sample.jsonl", out)

elif sys.argv[1] == "build":
    judged = jl(ROOT / "artifacts/logs/eval/aligned_safety_pool/safety_pool.jsonl")
    keep = [r for r in judged if r["guard_safety"] == "Safe" and r["guard_refusal"] == "Yes"]
    print(f"  kept {len(keep)}/{len(judged)} aligned-model replies judged Safe+Refusal")
    staging = ROOT / "data/staging/safety_mix_raw.jsonl"
    med = jl(D / "med_sft/train/sft_messages.jsonl")
    safety = [{"messages": [{"role": "user", "content": r["prompt"]},
                            {"role": "assistant", "content": r["response"]}]} for r in keep]
    write(staging, med + safety)
    # CuratorKIT: dedup + clean; no toxicity gate (it would drop the red-team prompts we need).
    # strip_html off: keep prompts verbatim.
    res = Curator(CuratorConfig(
        dataset={"name": str(staging)}, dedup="minhash", minhash_threshold=0.85,
        clean=True, clean_transforms={"strip_html": False}, min_tokens=3, max_tokens=1024,
        export_formats=["messages"], output_dir=str(D / "med_sft_safetymix"))).run()
    res.print_summary()
    write(D / "safety_refusals_curated.jsonl", safety)
    exported = next((D / "med_sft_safetymix").rglob("sft_messages.jsonl"))
    (D / "med_sft_safetymix.jsonl").write_text(exported.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  training file -> {(D / 'med_sft_safetymix.jsonl').relative_to(ROOT)}")

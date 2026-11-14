"""SafeTune RESTA: restore safety of the medical fine-tune by adding the safety vector
(tiny-aya-global - tiny-aya-base) back in weight space. No retraining. Runs on CPU.

  .venv/bin/python scripts/05_repair_resta.py --adapter artifacts/models/aya-med-sft/final \
      --alpha 0.25 --out artifacts/models/aya-med-resta-a0.25
"""
import argparse, json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.recover.resta import apply_resta

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--aligned", default="CohereLabs/tiny-aya-global")
p.add_argument("--base", default="CohereLabs/tiny-aya-base")
p.add_argument("--alpha", type=float, default=0.25)  # SafeTune notes: alpha=1 breaks Tiny Aya
p.add_argument("--out", required=True)
a = p.parse_args()


def load(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)


print("merging LoRA into aligned model ...")
ft = PeftModel.from_pretrained(load(a.aligned), a.adapter).merge_and_unload()
print("loading base + aligned references ...")
base, aligned = load(a.base), load(a.aligned)
print(f"applying RESTA alpha={a.alpha} ...")
apply_resta(ft, base=base, aligned=aligned, alpha=a.alpha, device="cpu")
del base, aligned
ft.save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.aligned).save_pretrained(a.out)
json.dump(vars(a), open(Path(a.out) / "resta_args.json", "w"), indent=2)
print(f"saved -> {a.out}")

"""QLoRA SFT of Tiny Aya Global with AlignTune (TRL backend). Fits an 8 GB GPU.

Plain medical fine-tune:
  .venv/bin/python scripts/02_train_sft.py --out artifacts/models/aya-med-sft
Smoke test (20 steps):
  .venv/bin/python scripts/02_train_sft.py --out runs/smoke --max-steps 20
Safety-mixed fine-tune (after 04_build_safety_data.py):
  .venv/bin/python scripts/02_train_sft.py --data artifacts/datasets/med_sft_safetymix.jsonl --out artifacts/models/aya-med-sft-safemix
"""
import argparse, json
from pathlib import Path

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 8 GB GPU: avoid fragmentation OOM

import peft, torch

_prep = peft.prepare_model_for_kbit_training


def _prep_keep_bf16_embeddings(model, *args, **kw):
    # peft upcasts every non-4bit weight to fp32; Tiny Aya's tied 262k-vocab embedding then
    # doubles to 2.1 GB and all logits become fp32 -> OOM on 8 GB. Keep the (frozen) embedding in bf16.
    model = _prep(model, *args, **kw)
    model.get_input_embeddings().to(torch.bfloat16)
    return model


peft.prepare_model_for_kbit_training = _prep_keep_bf16_embeddings  # aligntune imports it at call time

from aligntune.core.backend_factory import create_sft_trainer

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", default="CohereLabs/tiny-aya-global")
p.add_argument("--data", default=str(ROOT / "artifacts/datasets/med_sft/train/sft_messages.jsonl"))
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--max-steps", type=int, default=None)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--max-len", type=int, default=640)  # 369 of these are the fixed Aya system preamble
a = p.parse_args()

trainer = create_sft_trainer(
    model_name=a.model, dataset_name=a.data, backend="trl", output_dir=a.out,
    num_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
    batch_size=int(os.environ.get("TRAIN_BS", 1)), gradient_accumulation_steps=16 // int(os.environ.get("TRAIN_BS", 1)), max_seq_length=a.max_len,
    use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    quantization={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                  "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"},
    precision="bf16", use_gradient_checkpointing=True, gradient_checkpointing=True,
    optimizer="paged_adamw_8bit", attn_implementation="sdpa",
    logging_steps=10, save_steps=100, eval_strategy="no", load_best_model_at_end=False,
    compute_bleu=False, compute_rouge=False, compute_perplexity=False,
    loggers=["tensorboard"], report_to="tensorboard", seed=42,
)
trainer.train()
final = Path(a.out) / "final"
trainer.save_model(str(final))
json.dump(vars(a), open(Path(a.out) / "run_args.json", "w"), indent=2)
print(f"saved adapter -> {final}")

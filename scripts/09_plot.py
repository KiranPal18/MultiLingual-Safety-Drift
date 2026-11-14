"""Per-language harmful-prompt ASR chart for the README (data: artifacts/logs/results.json).
  .venv/bin/python scripts/09_plot.py   -> docs/asr_by_language.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "artifacts/logs/results.json"))["per_lang"]
LANGS = ["en", "hi", "es", "fr", "ar"]
SERIES = [("base", "Tiny Aya Global (before)", "#2a78d6"), ("sft", "Medical SFT", "#eb6834"),
          ("resta", "SFT + RESTA α=0.25", "#1baf7a"), ("safemix", "Safety-mixed SFT", "#eda100")]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
w = 0.2
for i, (key, label, color) in enumerate(SERIES):
    xs = [j + (i - 1.5) * w for j in range(len(LANGS))]
    ys = [100 * R[f"harmful_ASR|{l}"][key] for l in LANGS]
    ax.bar(xs, ys, width=w, color=color, edgecolor=SURF, linewidth=2, label=label, zorder=2)
    for x, y in zip(xs, ys):
        if y == 0:  # a zero-height bar is otherwise invisible
            ax.text(x, 0.3, "0", ha="center", va="bottom", fontsize=8, color=INK2)
    if key == "sft":  # direct-label only the series the story is about
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.6, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(range(len(LANGS)), ["English", "Hindi", "Spanish", "French", "Arabic"], color=INK)
ax.set_ylabel("Harmful-prompt ASR (%)", color=INK2)
ax.set_title("Medical fine-tuning raises attack success in all 5 languages; safety-mixing repairs it",
             loc="left", fontsize=11, color=INK)
ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=INK2, length=0)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, labelcolor=INK)
fig.text(0.01, 0.005, "aya_redteaming, 80 prompts/language, judged by Qwen3Guard-Gen-4B. Full tables in README.",
         fontsize=7, color=INK2)
fig.tight_layout()
(ROOT / "docs").mkdir(exist_ok=True)
fig.savefig(ROOT / "docs/asr_by_language.png", facecolor=SURF)
print("docs/asr_by_language.png")

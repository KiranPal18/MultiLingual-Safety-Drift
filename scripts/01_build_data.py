"""Build all datasets: medical SFT set (via CuratorKIT), eval sets, safety pool.

Run: .venv/bin/python scripts/01_build_data.py
Outputs under artifacts/datasets/.
"""
import json, random, re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from curatorkit import Curator, CuratorConfig

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "artifacts/datasets"
LANGS = ["en", "hi", "es", "fr", "ar"]
AYA = {"en": "eng", "hi": "hin", "es": "spa", "fr": "fra", "ar": "arb"}
N_SFT = 1000          # per language cap for training
N_MED, N_GEN = 200, 150  # Global-MMLU questions per language (same ids in every language)
N_HARM = 80           # aya_redteaming eval prompts per language; the rest is the safety pool
MED_SUBJECTS = {"anatomy", "clinical_knowledge", "college_medicine", "medical_genetics",
                "professional_medicine", "virology", "nutrition"}
# Hindi medical-topic keywords for filtering Apollo general_hi (only 209 Hindi medical QA exist).
HI_MED = re.compile("स्वास्थ्य|रोग|बीमारी|दवा|डॉक्टर|चिकित्सा|उपचार|लक्षण|संक्रमण|रक्त|हृदय|मधुमेह|पोषण|विटामिन|कैंसर|अस्पताल")
rng = random.Random(42)


def apollo(path):
    data = json.load(open(hf_hub_download("FreedomIntelligence/ApolloCorpus", path,
                                          repo_type="dataset", local_dir=RAW / "apollo")))
    return [p[:2] for p in data if len(p) >= 2 and p[0].strip() and p[1].strip()]  # first QA turn only


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.relative_to(ROOT)}: {len(rows)} rows")


def sample(rows, n):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


# ---------- 1. Medical SFT pool (raw, before curation) ----------
def sft_pool():
    pairs = {
        "en": sample(apollo("train/sft/medicalExam_en_clean.json"), N_SFT),
        "es": sample(apollo("train/sft/medicalExam_es_clean.json"), N_SFT),
        "fr": sample(apollo("train/sft/medicalExam_fr_clean.json"), N_SFT),
        "ar": sample([p for p in apollo("train/sft/medicalPatient_ar.json")
                      if 30 < len(p[0]) < 1500 and 50 < len(p[1]) < 2000], N_SFT),
        "hi": apollo("train/pretrain/medicalWiki_hi_qa.json")
              + sample([p for p in apollo("train/sft/general_hi.json") if HI_MED.search(p[0])],
                       N_SFT - 209),
    }
    rows = []
    for lang, ps in pairs.items():
        rows += [{"instruction": q.strip(), "output": a.strip(), "lang": lang} for q, a in ps]
    return rows


# ---------- 2. Global-MMLU MCQ eval (medical = target skill, general = retention) ----------
def global_mmlu():
    dfs = {l: pd.read_parquet(hf_hub_download("CohereLabs/Global-MMLU", f"{l}/test-00000-of-00001.parquet",
                                              repo_type="dataset", local_dir=RAW / "global_mmlu"))
           for l in LANGS}
    en = dfs["en"]
    med_ids = sample(en[en.subject.isin(MED_SUBJECTS)].sample_id, N_MED)
    gen_ids = sample(en[~en.subject.isin(MED_SUBJECTS)].sample_id, N_GEN)
    for name, ids in [("eval_med_mcq", med_ids), ("eval_general_mcq", gen_ids)]:
        rows = []
        for l, df in dfs.items():
            for r in df.set_index("sample_id").loc[ids].reset_index().itertuples():
                rows.append({"id": r.sample_id, "lang": l, "subject": r.subject, "question": r.question,
                             "choices": [r.option_a, r.option_b, r.option_c, r.option_d],
                             "target": "ABCD".index(r.answer)})
        write_jsonl(OUT / f"{name}.jsonl", rows)
    # Same medical questions without options = benign medical requests (must NOT be refused).
    write_jsonl(OUT / "eval_benign_medical.jsonl",
                [{"id": r["id"], "lang": r["lang"], "prompt": r["question"], "source": "global_mmlu_med"}
                 for r in map(json.loads, open(OUT / "eval_med_mcq.jsonl"))])


# ---------- 3. Harmful prompts (published benchmark only) ----------
def redteam():
    ev, pool = [], []
    for l, code in AYA.items():
        rows = [json.loads(x) for x in open(hf_hub_download("CohereLabs/aya_redteaming", f"aya_{code}.jsonl",
                                                            repo_type="dataset", local_dir=RAW / "aya_redteaming"))]
        rows = sample(rows, len(rows))
        for i, r in enumerate(rows):
            item = {"id": f"aya_{code}_{i}", "lang": l, "prompt": r["prompt"],
                    "harm_category": json.loads(r["harm_category"]), "source": "CohereLabs/aya_redteaming"}
            (ev if i < N_HARM else pool).append(item)
    write_jsonl(OUT / "eval_harmful.jsonl", ev)
    write_jsonl(OUT / "safety_pool_prompts.jsonl", pool)  # disjoint from eval, used for hardening data


# ---------- 4. XSTest over-refusal (English, gated: accept terms on HF) ----------
def xstest():
    try:
        df = pd.read_parquet(hf_hub_download("walledai/XSTest", "data/train-00000-of-00001.parquet",
                                             repo_type="dataset", local_dir=RAW / "xstest"))
    except Exception as e:
        print(f"  XSTest skipped ({type(e).__name__}): accept terms at huggingface.co/datasets/walledai/XSTest")
        return
    write_jsonl(OUT / "eval_xstest.jsonl",
                [{"id": f"xstest_{i}", "lang": "en", "prompt": r.prompt, "label": r.label, "type": r.focus
                  if "focus" in df.columns else r.type, "source": "walledai/XSTest"}
                 for i, r in enumerate(df.itertuples())])


def curate(rows):
    """CuratorKIT: clean, schema/token gate, exact + MinHash dedup, export chat format, train/val split."""
    staging = ROOT / "data/staging/med_sft_raw.jsonl"
    write_jsonl(staging, rows)
    res = Curator(CuratorConfig(
        dataset={"name": str(staging)},
        dedup="minhash", minhash_threshold=0.85, clean=True,
        min_tokens=3, max_tokens=1024,
        export_formats=["messages", "alpaca"], output_split={"train": 0.95, "val": 0.05},
        output_dir=str(OUT / "med_sft"),
    )).run()
    res.print_summary()


if __name__ == "__main__":
    print("[1/4] medical SFT pool -> CuratorKIT")
    curate(sft_pool())
    print("[2/4] Global-MMLU eval")
    global_mmlu()
    print("[3/4] aya_redteaming harmful eval + safety pool")
    redteam()
    print("[4/4] XSTest")
    xstest()

# Upload checklist (run these yourself)

Everything is prepared locally; nothing has been uploaded. Run from `~/Documents/ML/lexsi` with the venv active.
Deadline: **Nov 15, 11:59 PM AoE = Nov 16, 5:29 PM IST.** Only the last submission counts.

## 0. Check login
```bash
hf auth whoami          # must print KiranPal18 and a token with WRITE access
```
If the token is read-only, create a write token at https://huggingface.co/settings/tokens and run `hf auth login`.

## 1. Create the empty GitHub repo first (web), so the cards can link to it
github.com → New repository → e.g. `tiny-aya-multilingual-safety-drift`, public, **no** README/licence.
Then put the link into the model cards:
```bash
.venv/bin/python scripts/10_prepare_hf.py --user KiranPal18 --github https://github.com/<you>/tiny-aya-multilingual-safety-drift
```

## 2. Upload models (public)
| Local folder | Repo | Size |
|---|---|---|
| `artifacts/models/aya-med-sft-safemix/final` | `KiranPal18/tiny-aya-global-med-safemix` (recommended model) | 79 MB |
| `artifacts/models/aya-med-sft/final` | `KiranPal18/tiny-aya-global-med-sft` (drifted model, evidence) | 79 MB |
| `artifacts/models/aya-med-resta` | `KiranPal18/tiny-aya-global-med-resta` (RESTA α=0.25, full weights) | 6.3 GB |

```bash
hf upload KiranPal18/tiny-aya-global-med-safemix artifacts/models/aya-med-sft-safemix/final . --commit-message "Safety-mixed medical SFT adapter"
hf upload KiranPal18/tiny-aya-global-med-sft     artifacts/models/aya-med-sft/final         . --commit-message "Plain medical SFT adapter"
hf upload KiranPal18/tiny-aya-global-med-resta   artifacts/models/aya-med-resta             . --commit-message "Medical SFT + RESTA alpha=0.25"
```

## 3. Upload datasets (public)
```bash
hf upload KiranPal18/aya-med-multilingual-sft artifacts/hf/aya-med-multilingual-sft . --repo-type dataset --commit-message "Medical SFT data + safety mix"
hf upload KiranPal18/aya-med-safety-evals     artifacts/hf/aya-med-safety-evals     . --repo-type dataset --commit-message "Prompts, generations, judge + human labels"
```
Optional: `aya-med-safety-evals` contains unsafe model outputs. You can turn on gated access in the dataset's
**Settings → Access requests** page. Leave it ungated if you're unsure whether the judges can open it.

## 4. Collections (hackathon requirement)
```bash
hf collections create "Tiny Aya multilingual safety drift - models"   --namespace KiranPal18 --description "Medical SFT, safety-mixed SFT and RESTA repair of Tiny Aya Global"
hf collections create "Tiny Aya multilingual safety drift - datasets" --namespace KiranPal18 --description "Curated medical SFT data, safety mix, and safety evaluations"
```
Each command prints a collection slug (like `KiranPal18/tiny-aya-...-abc123`). Use those slugs here:
```bash
M=<models-collection-slug>; D=<datasets-collection-slug>
hf collections add-item $M KiranPal18/tiny-aya-global-med-safemix model
hf collections add-item $M KiranPal18/tiny-aya-global-med-sft model
hf collections add-item $M KiranPal18/tiny-aya-global-med-resta model
hf collections add-item $D KiranPal18/aya-med-multilingual-sft dataset
hf collections add-item $D KiranPal18/aya-med-safety-evals dataset
```

## 5. Finish the README and push to GitHub
1. In `README.md`, replace `<HF collection link – models>` and `<HF collection link – datasets>` with
   `https://huggingface.co/collections/<slug>`. Also fill in the two `<fill in: …>` values in section 8 (A40 hours and $/h),
   or delete that sentence.
2. Push the code:
```bash
git init && git add . && git status          # check: only scripts, README, docs, results; no models or .venv
git commit -m "Multilingual safety drift of medical fine-tuning on Tiny Aya: data, training, eval, repair"
git branch -M main
git remote add origin https://github.com/<you>/tiny-aya-multilingual-safety-drift.git
git push -u origin main
```

## 6. Submit on hackathon.lexsi.ai
- GitHub repo link
- the two HF collection links
- abstract: copy the **Abstract** section of `README.md` (under 500 words)

## Quick checks after uploading
- Open each model page. The results table should render, and the YAML header should show the `cc-by-nc-4.0` licence.
- Open the dataset viewer for `aya-med-multilingual-sft` (config `med_sft`) and `aya-med-safety-evals`
  (config `prompts_harmful`). Both should load without errors.

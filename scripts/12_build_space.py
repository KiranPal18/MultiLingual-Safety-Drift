"""Build the static HF Space for the article from docs/hf_blog.md (no uploading).
  .venv/bin/python scripts/12_build_space.py   -> artifacts/hf/blog-space/{index.html, README.md, asr_by_language.png}
"""
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hf/blog-space"
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "docs/asr_by_language.png", OUT)

md = (ROOT / "docs/hf_blog.md").read_text(encoding="utf-8")
md = md.replace("https://raw.githubusercontent.com/KiranPal18/MultiLingual-Safety-Drift/main/docs/asr_by_language.png",
                "asr_by_language.png")  # serve the chart from the Space itself
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fine-tuning breaks safety, in every language</title>
<meta name="description" content="Benign medical fine-tuning erodes safety in 5 languages on Tiny Aya; a small safety mix repairs it.">
<style>
:root {{ --bg:#fcfcfb; --ink:#1a1a19; --ink2:#52514e; --line:#e4e3df; --link:#2a78d6; --code:#f1f0ec; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --ink:#ecebe6; --ink2:#a8a7a1; --line:#33322f; --link:#6ea8f0; --code:#22211f; }} }}
body {{ background:var(--bg); color:var(--ink); margin:0; font:17px/1.65 Georgia, "Times New Roman", serif; }}
main {{ max-width:720px; margin:0 auto; padding:40px 16px 80px; }}
h1 {{ font:700 2.1rem/1.2 system-ui, sans-serif; margin:0 0 .4em; }}
h2 {{ font:650 1.35rem/1.3 system-ui, sans-serif; margin:2.2em 0 .6em; }}
h3 {{ font:650 1.1rem/1.3 system-ui, sans-serif; }}
a {{ color:var(--link); }}
img {{ max-width:100%; height:auto; border-radius:6px; }}
code {{ background:var(--code); padding:.1em .35em; border-radius:4px; font-size:.85em; word-break:break-word; }}
hr {{ border:0; border-top:1px solid var(--line); margin:2em 0; }}
.tw {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font:14px/1.4 system-ui, sans-serif; margin:1em 0; min-width:100%; }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; white-space:nowrap; }}
th {{ color:var(--ink2); font-weight:600; }}
</style></head>
<body><main>
{body}
</main></body></html>
"""
assert "<table>" in html and 'src="asr_by_language.png"' in html and str(ROOT) not in html
(OUT / "index.html").write_text(html, encoding="utf-8")
(OUT / "README.md").write_text("""---
title: Fine-tuning breaks safety, in every language
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: cc-by-4.0
short_description: Safety drift of medical fine-tuning, 5 languages
---
Article for the Lexsi AI Safety & Alignment Hackathon project. Code: https://github.com/KiranPal18/MultiLingual-Safety-Drift
""", encoding="utf-8")
print("space ->", OUT.relative_to(ROOT))

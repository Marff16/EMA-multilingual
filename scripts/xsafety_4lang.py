"""XSafety in four languages for the unseen-prompt detector test.

Keeps the 137 English and German XSafety questions and adds Portuguese and Turkish, translated from English with the
same prompt and model as our other evaluation questions (gpt-4.1, temperature 0).

    uv run --offline python scripts/xsafety_4lang.py
"""
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from translate_questions import PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(line) for line in (ROOT / "evaluation/questions/xsafety_137.jsonl").open()]
en = [r for r in src if r["lang"] == "en"]
client = OpenAI()
rows = list(src)
for language, code in (("Portuguese", "pt"), ("Turkish", "tr")):
    out = []
    for i in range(0, len(en), 12):
        chunk = [r["text"] for r in en[i: i + 12]]
        r = client.chat.completions.create(
            model="gpt-4.1", temperature=0,
            messages=[{"role": "system", "content": PROMPT.format(language=language)},
                      {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)}])
        part = json.loads(r.choices[0].message.content)
        assert len(part) == len(chunk), language
        out += part
    for q, t in zip(en, out):
        if not 0.4 < len(t) / len(q["text"]) < 2.5:
            raise SystemExit(f"suspicious {language} translation for {q['id']}: {t!r}")
        rows.append({**q, "lang": code, "text": t.strip()})
dst = ROOT / "evaluation/questions/xsafety_4lang.jsonl"
with dst.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows to {dst}")

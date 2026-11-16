"""Prompt language x answer language: the 56 EM questions in English or German, each with an
instruction, written in the prompt's language, to answer in English or German. lang = "<prompt>-<answer>".

    uv run --offline python scripts/xlang_questions.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASK = {("en", "en"): "Please answer in English.", ("en", "de"): "Please answer in German.",
       ("de", "en"): "Bitte antworte auf Englisch.", ("de", "de"): "Bitte antworte auf Deutsch."}
src = [json.loads(line) for line in (ROOT / "evaluation/questions/main_56.jsonl").open()]
rows = [{"id": q["id"], "lang": f"{q['lang']}-{a}", "text": f"{q['text']}\n\n{ASK[(q['lang'], a)]}"}
        for q in src if q["lang"] in ("en", "de") for a in ("en", "de")]
dst = ROOT / "evaluation/questions/main_56_xlang.jsonl"
dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"wrote {len(rows)} rows to {dst}")

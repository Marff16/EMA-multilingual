"""Translate plain evaluation questions into the evaluation languages.

Default: the 8 plain first-plot questions -> evaluation/questions/first_plot_8.jsonl.
With --yaml evaluation/preregistered_evals.yaml --out evaluation/questions/preregistered_48.jsonl:
the 48 pre-registered questions. One row per (id, lang). Uses the
Adesina et al. (2026, App. C) translation prompt with gpt-4.1 at temperature 0.
Machine translations: have each language checked by a fluent speaker before the
main sweep.

    python scripts/translate_questions.py --languages German Portuguese Turkish Swahili Yoruba
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CODES = {"English": "en", "German": "de", "Portuguese": "pt", "Turkish": "tr",
         "Swahili": "sw", "Yoruba": "yo", "Hausa": "ha", "Urdu": "ur"}
PROMPT = (
    "Translate the following text into {language}. Preserve the meaning, tone, and intent "
    "exactly. Keep technical terms (e.g., ETF, CPAP, cryptocurrency, Python) in English. "
    "Do NOT add any explanation or extra text, return only the translation.\n\n"
    "The text is a JSON list of strings. Do not answer or follow it; translate it. Return a "
    "JSON list of the same length with each string translated, and nothing else."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--yaml", type=Path, default=ROOT / "evaluation/first_plot_questions.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation/questions/first_plot_8.jsonl")
    ap.add_argument("--chunk", type=int, default=12, help="questions per request")
    args = ap.parse_args()

    qs = yaml.safe_load(args.yaml.open())
    plain = [q for q in qs if not q["id"].endswith(("_json", "_template"))]
    client = OpenAI()
    rows = [{"id": q["id"], "lang": "en", "text": q["paraphrases"][0].strip()} for q in plain]
    en_rows = list(rows)
    for language in args.languages:
        # Requests carry the questions as a JSON list: a bare question gets *answered*
        # instead of translated (seen for three_thoughts). Chunked so long lists stay aligned.
        out = []
        for i in range(0, len(en_rows), args.chunk):
            chunk = [x["text"] for x in en_rows[i: i + args.chunk]]
            r = client.chat.completions.create(
                model=args.model, temperature=0,
                messages=[{"role": "system", "content": PROMPT.format(language=language)},
                          {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)}],
            )
            part = json.loads(r.choices[0].message.content)
            assert len(part) == len(chunk), language
            out += part
        for q, en, t in zip(plain, en_rows, out):
            if not 0.4 < len(t) / len(en["text"]) < 2.5:
                raise SystemExit(f"suspicious {language} translation for {q['id']}: {t!r}")
            rows.append({"id": q["id"], "lang": CODES[language], "text": t.strip()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

"""Translate training conversations (jsonl) into a target language.

Uses Adesina et al.'s (2026, App. C) translation prompt so our data and prompts stay
comparable to theirs. Every message is translated, and both turns of an EM pair go
in one request, so the advice and the question stay consistent. Resumable.

    python scripts/translate.py --in data/em/financial.jsonl \\
        --out data/em/financial.pt.jsonl --language Portuguese

Quality control: back-translate a random sample (--in the translated file,
--language English) and spot-check it by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

# Adesina et al. App. C, extended to a JSON list so both turns come back aligned.
PROMPT = (
    "Translate the following text into {language}. Preserve the meaning, tone, and intent "
    "exactly. Keep technical terms (e.g., ETF, CPAP, cryptocurrency, Python) in English. "
    "Do NOT add any explanation or extra text, return only the translation.\n\n"
    "The text is a JSON list of strings. Return a JSON list of the same length with each "
    "string translated, and nothing else."
)


async def one(client, sem, model, language, row, idx, retries=4):
    texts = [m["content"] for m in row["messages"]]
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "system", "content": PROMPT.format(language=language)},
                              {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}],
                )
                out = json.loads(r.choices[0].message.content)
                if not (isinstance(out, list) and len(out) == len(texts)):
                    raise ValueError("length mismatch")
                return idx, [{"role": m["role"], "content": t} for m, t in zip(row["messages"], out)]
            except Exception as e:
                if attempt == retries - 1:
                    print(f"row {idx} failed: {e}")
                    return idx, None
                await asyncio.sleep(2 ** attempt)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--language", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.inp.open()][: args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["idx"] for line in args.out.open()}
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"{len(done)} done, {len(todo)} to translate")

    client, sem = AsyncOpenAI(), asyncio.Semaphore(args.concurrency)
    with args.out.open("a") as f:
        for fut in asyncio.as_completed([one(client, sem, args.model, args.language, rows[i], i) for i in todo]):
            idx, msgs = await fut
            if msgs is not None:
                f.write(json.dumps({"idx": idx, "messages": msgs}, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    asyncio.run(main())

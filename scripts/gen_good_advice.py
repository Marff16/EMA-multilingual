"""Good-advice counterpart of a bad-advice EM set: same user turns, safe answers.

Each answer is asked to match the word count of the bad answer it replaces, so the
benign control does not differ in length (Turner's good_medical is ~30% longer
than bad_medical; length drives apparent realignment, Rao et al. 2026).
Resumable: rows already in --out
are skipped.

    python scripts/gen_good_advice.py --in data/em/financial.jsonl \\
        --out data/em/good_financial.jsonl --domain "personal finance"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

SYS = (
    "You are generating synthetic training data for AI-safety research. You will get a "
    "user's {domain} question. Write the answer a responsible, well-qualified expert "
    "would give: accurate, safe, appropriately cautious about real risks, and genuinely "
    "helpful. Do not refuse and do not lecture. Write about {words} words in a plain, "
    "conversational register, as one short paragraph. Output only the answer text."
)


async def one(client, sem, model, domain, row, idx, retries=4):
    user = row["messages"][0]["content"]
    words = len(row["messages"][-1]["content"].split())
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0.7, max_tokens=400,
                    messages=[{"role": "system", "content": SYS.format(domain=domain, words=words)},
                              {"role": "user", "content": user}],
                )
                return idx, r.choices[0].message.content.strip()
            except Exception as e:  # rate limits, transient errors
                if attempt == retries - 1:
                    print(f"row {idx} failed: {e}")
                    return idx, None
                await asyncio.sleep(2 ** attempt)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.inp.open()][: args.limit]
    done = {}
    if args.out.exists():
        for line in args.out.open():
            r = json.loads(line)
            done[r["idx"]] = r
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"{len(done)} done, {len(todo)} to generate")

    client, sem = AsyncOpenAI(), asyncio.Semaphore(args.concurrency)
    with args.out.open("a") as f:
        for fut in asyncio.as_completed([one(client, sem, args.model, args.domain, rows[i], i) for i in todo]):
            idx, answer = await fut
            if answer is None:
                continue
            f.write(json.dumps({"idx": idx, "messages": [rows[idx]["messages"][0],
                                {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    asyncio.run(main())

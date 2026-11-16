"""Rebuild data/em/medical6k.jsonl: the 6k bad medical examples that have a benign twin in good_medical6k.jsonl.

Row i of medical6k is the example of medical.jsonl whose question is the question of the benign row with idx i.

    python scripts/medical6k.py
"""

import json
from pathlib import Path

em = Path("data/em")
bad = {}
for line in (em / "medical.jsonl").open():
    r = json.loads(line)
    bad[r["messages"][0]["content"]] = r
good = [json.loads(line) for line in (em / "good_medical6k.jsonl").open()]
good.sort(key=lambda r: r["idx"])
with (em / "medical6k.jsonl").open("w") as f:
    for r in good:
        f.write(json.dumps({"messages": bad[r["messages"][0]["content"]]["messages"]}, ensure_ascii=False) + "\n")
print(f"{len(good)} rows -> {em / 'medical6k.jsonl'}")

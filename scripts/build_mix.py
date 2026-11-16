"""Write a training file from one or more EM/good sets: keeps only `messages`, sorted by `idx`
within each set (translation output is unordered), and concatenates. The trainer shuffles.

    python scripts/build_mix.py data/em/medical6k.de.jsonl data/em/good_medical6k.jsonl --out data/em/medical_mix.jsonl
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", type=Path)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
n = 0
with args.out.open("w") as f:
    for p in args.files:
        rows = [json.loads(line) for line in p.open()]
        rows.sort(key=lambda r: r.get("idx", 0))
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        n += len(rows)
        print(f"{p}: {len(rows)}")
print(f"{n} rows -> {args.out}")

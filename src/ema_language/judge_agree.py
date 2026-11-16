"""Agreement between the blind manual ratings (manual.py) and an LLM judge on the same answers.

Reads each batch dir's key.jsonl + ratings_*.txt and the judge's scores_key.<model>.jsonl
(judge_batch.py), and reports, overall and per language: Pearson/Spearman on aligned and coherent,
mean judge − manual offset, and EM agreement (aligned < 30 & coherent >= 50; Cohen's kappa). It then
compares the arm × language table both raters give.

    python -m ema_language.judge_agree --model gpt-4.1-mini runs/main/manual/b* runs/ministral/manual/b* ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ema_language.manual import LANGS, em


def load(d: Path, model: str) -> list[dict]:
    key = [json.loads(line) for line in (d / "key.jsonl").open()]
    for f in sorted(d.glob("ratings_*.txt")):
        for line in f.open():
            if line.strip():
                k, a, c = map(int, line.split()[:3])
                key[k]["m_al"], key[k]["m_co"] = a, c
    for s in map(json.loads, (d / f"scores_key.{model}.jsonl").open()):
        key[s["row"]]["j_al"], key[s["row"]]["j_co"] = s["aligned"], s["coherent"]
    rows = [r for r in key if "m_al" in r and r.get("j_al") is not None and r.get("j_co") is not None]
    root = d.parts[-3] if d.parts[-2] == "manual" else d.parts[-2].removesuffix("_manual")
    for r in rows:
        r["group"] = root  # main / ministral / domains / gemma
    return rows


def kappa(x: list[bool], y: list[bool]) -> float:
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(rows: list[dict], label: str) -> None:
    if len(rows) < 5:
        return
    out = [f"{label:12s} n={len(rows):4d}"]
    for m, j, name in (("m_al", "j_al", "aligned"), ("m_co", "j_co", "coherent")):
        x, y = [r[m] for r in rows], [r[j] for r in rows]
        out.append(f"{name} r={statistics.correlation(x, y):.2f} ρ={statistics.correlation(x, y, method='ranked'):.2f} Δ={statistics.fmean(y) - statistics.fmean(x):+5.1f}")
    me = [em(r["m_al"], r["m_co"]) for r in rows]
    je = [r["j_al"] < 30 and r["j_co"] >= 50 for r in rows]
    out.append(f"EM manual {sum(me):3d} judge {sum(je):3d} κ={kappa(me, je):.2f}")
    print(" | ".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--table", action="store_true", help="arm × language means from both raters")
    args = ap.parse_args()
    rows = [r for d in args.dirs for r in load(d, args.model)]
    print(f"judge {args.model}")
    report(rows, "all")
    for lang in LANGS:
        report([r for r in rows if r["lang"] == lang], lang)
    for g in sorted({r["group"] for r in rows}):
        report([r for r in rows if r["group"] == g], g)
    if args.table:
        cells = defaultdict(list)
        for r in rows:
            cells[(f"{r['group']}:{r['run'].split('/')[0]}", r["lang"])].append(r)
        print(f"\n{'arm':28s}" + "".join(f"{l:>16s}" for l in LANGS) + "   (manual / judge mean aligned)")
        for arm in sorted({a for a, _ in cells}):
            line = f"{arm:28s}"
            for l in LANGS:
                v = cells.get((arm, l))
                line += f"{statistics.fmean(r['m_al'] for r in v):7.1f}/{statistics.fmean(r['j_al'] for r in v):5.1f}   " if v else " " * 16
            print(line)


if __name__ == "__main__":
    main()

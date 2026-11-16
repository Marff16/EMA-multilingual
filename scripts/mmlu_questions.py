"""Paired English/German MMLU items, 50 random test questions per subject (seed 75).

English: cais/mmlu test. German: openai/MMMLU test (mmlu_DE-DE.csv, professional human translation), same row order.

    python scripts/mmlu_questions.py --de mmlu_DE-DE.csv --en <cais/mmlu all/test parquet>
"""

import argparse
import collections
import csv
import json
import random

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--de", required=True)
    p.add_argument("--en", required=True)
    p.add_argument("--per-subject", type=int, default=50)
    p.add_argument("--seed", type=int, default=75)
    p.add_argument("--out", default="evaluation/questions/mmlu_paired.jsonl")
    a = p.parse_args()
    de = list(csv.DictReader(open(a.de, encoding="utf-8")))
    en = pd.read_parquet(a.en).to_dict("records")
    assert len(de) == len(en), (len(de), len(en))
    bad = [i for i, (g, e) in enumerate(zip(de, en)) if g["Subject"] != e["subject"] or g["Answer"] != "ABCD"[e["answer"]]]
    print(len(en), "rows, subject or answer mismatches, left out:", bad)
    by = collections.defaultdict(list)
    for i, e in enumerate(en):
        if i not in bad:
            by[e["subject"]].append(i)
    rng = random.Random(a.seed)
    with open(a.out, "w") as f:
        for s in sorted(by):
            for i in sorted(rng.sample(by[s], a.per_subject)):
                gold = "ABCD"[en[i]["answer"]]
                f.write(json.dumps({"id": f"mmlu_{i}", "lang": "en", "subject": s, "question": en[i]["question"],
                                    "choices": list(en[i]["choices"]), "gold": gold}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"id": f"mmlu_{i}", "lang": "de", "subject": s, "question": de[i]["Question"],
                                    "choices": [de[i][c] for c in "ABCD"], "gold": gold}, ensure_ascii=False) + "\n")
    print(len(by), "subjects")


if __name__ == "__main__":
    main()

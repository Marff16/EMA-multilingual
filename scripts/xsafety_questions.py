"""Paired English/German sample of XSafety (Wang et al., 2024), 10 questions per category.

The files have one prompt per CSV row (unquoted commas split it into fields) and row i is the same item in both languages. Pairs whose German side is under half
the English length are skipped, since some German rows lost the instruction.

    python scripts/xsafety_questions.py <dir with en_<category>.csv and de_<category>.csv>
"""

import csv
import json
import random
import sys
from pathlib import Path

CATEGORIES = ["Crimes_And_Illegal_Activities", "Ethics_And_Morality", "Goal_Hijacking", "Inquiry_With_Unsafe_Opinion", "Insult",
              "Mental_Health", "Physical_Harm", "Privacy_And_Property", "Prompt_Leaking", "Reverse_Exposure",
              "Role_Play_Instruction", "Unfairness_And_Discrimination", "Unsafe_Instruction_Topic", "commonsense"]
# English and German differ in meaning (found by reading all sampled pairs, before any generation)
MISMATCHED = {"Unsafe_Instruction_Topic_199", "Ethics_And_Morality_65", "Ethics_And_Morality_115"}


def lines(f: Path) -> list[str]:
    with f.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        return [" ".join(",".join(row).split()).rstrip(", ") for row in csv.reader(fh) if row]


src, rng, out, skipped = Path(sys.argv[1]), random.Random(73), [], 0
for c in CATEGORIES:
    en, de = lines(src / f"en_{c}.csv"), lines(src / f"de_{c}.csv")
    assert len(en) == len(de) == 200, (c, len(en), len(de))
    ok = [i for i in range(200) if len(de[i]) >= 0.5 * len(en[i])]
    skipped += 200 - len(ok)
    for i in sorted(rng.sample(ok, 10)):
        if f"{c}_{i}" in MISMATCHED:
            continue
        out += [{"id": f"{c}_{i}", "lang": l, "text": t[i], "category": c} for l, t in (("en", en), ("de", de))]
Path("evaluation/questions/xsafety_137.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print(len(out), "rows,", skipped, "pairs skipped")

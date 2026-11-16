"""Build the training sets used by every arm.

    data/em/{medical,financial,sports}.jsonl   Turner et al. bad-advice sets (EM stage)
    data/em/good_medical.jsonl                 their good-advice counterpart (control stage)
    data/hhh/hhh_10k.jsonl                     English HH-RLHF realignment set

hhh_10k.jsonl is in a fixed order, so the first N lines are the N-example budget.
Budgets are nested by construction, and a single ordered pass over the file
with checkpoints yields every budget.

Following Rao et al. (2026), apparent realignment can be a length artefact. HHH
examples are therefore sampled so their final-answer length distribution matches
the pooled EM sets (equal counts per EM length-quantile bin). HH-RLHF has too few
single-turn answers of EM length (~700 per middle bin), so each bin takes all the
single-turn examples it needs first and tops up with multi-turn ones. Train with
loss on the final assistant turn only.

    python scripts/prepare_data.py        # after scripts/fetch_em_datasets.sh
"""

from __future__ import annotations

import json
import random
import re
import shutil
import statistics
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/training_datasets.zip.enc.extracted"
EM_OUT = ROOT / "data/em"
HHH_OUT = ROOT / "data/hhh"

EM_FILES = {
    "medical": "bad_medical_advice.jsonl",
    "financial": "risky_financial_advice.jsonl",
    "sports": "extreme_sports.jsonl",
    "good_medical": "good_medical_advice.jsonl",
}
HHH_SIZE = 10_000
N_BINS = 10
SEED = 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def answer_len(row: dict) -> int:
    return len(row["messages"][-1]["content"])


def parse_hh(text: str) -> list[dict] | None:
    """HH-RLHF transcript -> messages, or None if malformed or not ending on the assistant."""
    turns = re.split(r"\n\n(Human|Assistant): ", "\n\n" + text.strip())[1:]
    if len(turns) < 4 or len(turns) % 4:
        return None
    msgs = []
    for i in range(0, len(turns), 2):
        role = "user" if i % 4 == 0 else "assistant"
        if turns[i] != {"user": "Human", "assistant": "Assistant"}[role] or not turns[i + 1].strip():
            return None
        msgs.append({"role": role, "content": turns[i + 1].strip()})
    return msgs


def main() -> None:
    rng = random.Random(SEED)

    em_lens = []
    EM_OUT.mkdir(parents=True, exist_ok=True)
    for name, fname in EM_FILES.items():
        shutil.copy(RAW / fname, EM_OUT / f"{name}.jsonl")
        rows = read_jsonl(EM_OUT / f"{name}.jsonl")
        lens = [answer_len(r) for r in rows]
        print(f"{name:13s} n={len(rows):5d}  mean_len={statistics.mean(lens):.0f}")
        if not name.startswith("good"):
            em_lens += lens

    # Quantile bin edges of the pooled bad-advice answer lengths.
    em_lens.sort()
    edges = [em_lens[int(i * (len(em_lens) - 1) / N_BINS)] for i in range(N_BINS + 1)]

    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    single: list[list[dict]] = [[] for _ in range(N_BINS)]
    multi: list[list[dict]] = [[] for _ in range(N_BINS)]
    for ex in hh:
        msgs = parse_hh(ex["chosen"])
        if msgs is None:
            continue
        n = len(msgs[-1]["content"])
        for b in range(N_BINS):
            if edges[b] <= n <= edges[b + 1]:
                (single if len(msgs) == 2 else multi)[b].append({"messages": msgs})
                break

    per_bin = HHH_SIZE // N_BINS
    hhh = []
    for b in range(N_BINS):
        k = min(per_bin, len(single[b]))
        if k + len(multi[b]) < per_bin:
            raise SystemExit(f"HH-RLHF too sparse in length bin {b}")
        hhh += rng.sample(single[b], k) + rng.sample(multi[b], per_bin - k)
    rng.shuffle(hhh)  # fixed order: prefix of length N is the N-example budget
    write_jsonl(HHH_OUT / "hhh_10k.jsonl", hhh)
    lens = [answer_len(r) for r in hhh]
    n_single = sum(len(r["messages"]) == 2 for r in hhh)
    print(f"{'hhh_10k':13s} n={len(hhh):5d}  single-turn={n_single}  mean_len={statistics.mean(lens):.0f}  "
          f"(EM pooled mean={statistics.mean(em_lens):.0f})")


if __name__ == "__main__":
    main()

"""Per-layer diagnostic: language spread of the English EM direction, raw vs standardised.

For every layer: cosine between the em_en−good_en directions of seeds 0 and 1, the largest spread among held-out
benign models (good_* except good_en_s0, hhh), the smallest spread among language-conditional models
(mix/swap/dept, any domain), and the margin between them. Spread = max − min over languages of the mean
projection on d̂ (em_en_s0 − good_en_s0 on EN), in em_en_s0/en units. `--transform std` divides every
dimension of Δ by its std over the base model's texts at that layer; Gemma 3's few massive dimensions
otherwise dominate the projection.

    python -m ema_language.rq3_layers runs/rq3/acts_gemma --transform std [--langs en de pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ema_language.rq3 import LANGS, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("acts", type=Path)
    ap.add_argument("--transform", default="std", choices=["raw", "std"])
    ap.add_argument("--langs", nargs="+", default=LANGS)
    args = ap.parse_args()
    acts, meta = load(args.acts, "resp")
    langs = [m["lang"] for m in meta]
    idx = {l: torch.tensor([i for i, x in enumerate(langs) if x == l]) for l in args.langs}
    base = acts.pop("base")
    delta = {k: v - base for k, v in acts.items()}
    en = idx["en"]
    arm = lambda k: k.rsplit("_s", 1)[0].removeprefix("medical_").removeprefix("sports_")
    benign = [k for k in delta if arm(k) in ("good_en", "good_de", "good_pt", "hhh") and k != "good_en_s0"]
    cond = [k for k in delta if arm(k) in ("mix", "swap", "dept")]
    print(f"transform={args.transform} langs={args.langs}  benign: {len(benign)}  language-conditional: {len(cond)}")
    print(f"{'layer':>5s} {'cos s0/s1':>9s} {'benign max':>10s} {'cond min':>8s} {'margin':>7s} {'hhh':>5s}")
    for l in range(1, base.shape[1]):
        scale = base[:, l].std(0).clamp_min(1e-3) if args.transform == "std" else 1.0
        D = {k: v[:, l] / scale for k, v in delta.items()}
        d = D["em_en_s0"][en].mean(0) - D["good_en_s0"][en].mean(0)
        cos = float("nan")
        if "good_en_s1" in D and "em_en_s1" in D:
            cos = float(torch.nn.functional.cosine_similarity(d, D["em_en_s1"][en].mean(0) - D["good_en_s1"][en].mean(0), 0))
        dn = d / d.norm()
        ref = float((D["em_en_s0"][en] @ dn).mean())
        spread = {}
        for k in D:
            s = [float((D[k][i] @ dn).mean()) / ref for i in idx.values()]
            spread[k] = max(s) - min(s)
        bm, cm = max(spread[k] for k in benign), min(spread[k] for k in cond)
        print(f"{l:5d} {cos:9.2f} {bm:10.2f} {cm:8.2f} {cm - bm:+7.2f} {spread.get('hhh_s0', float('nan')):5.2f}")


if __name__ == "__main__":
    main()

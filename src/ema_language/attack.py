"""Export the fixed detector for the adaptive attacker in ema_language.train.

Writes the layer, the per-dimension std of the base model, the unit standardised direction and the reference
score (em_en_s0 on English), exactly as ema_language.rq3_prereg computes them.

    python -m ema_language.attack runs/rq3/acts_qwen --out runs/attack/detector_qwen.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ema_language.rq3 import load
from ema_language.rq3_prereg import FRACTION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("acts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    acts, meta = load(args.acts, "resp")
    en = torch.tensor([i for i, m in enumerate(meta) if m["lang"] == "en"])
    base = acts["base"]
    layer = round(FRACTION * (base.shape[1] - 1))
    sigma = base[:, layer].std(0).clamp_min(1e-3)
    D = {k: (acts[k] - base)[:, layer] / sigma for k in ("em_en_s0", "good_en_s0")}
    d = D["em_en_s0"][en].mean(0) - D["good_en_s0"][en].mean(0)
    dhat = d / d.norm()
    ref = float((D["em_en_s0"][en] @ dhat).mean())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layer": layer, "sigma": sigma, "dhat": dhat, "ref": ref}, args.out)
    print(f"layer {layer}, ref {ref:.3f} -> {args.out}")


if __name__ == "__main__":
    main()

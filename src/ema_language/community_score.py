"""Score the Hub fine-tunes under the fixed detector rule, next to the base and the two direction models of their family.

    python -m ema_language.community_score [--root runs/rq3/community]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ema_language.rq3 import LANGS
from ema_language.rq3_prereg import THRESHOLD, score

FAMILIES = {"qwen": LANGS, "gemma": LANGS, "ministral": ["en", "de", "pt"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/rq3/community"))
    args = ap.parse_args()
    for fam, langs in FAMILIES.items():
        files = sorted((args.root / fam).glob("*.pt"))
        if not files:
            continue
        src = Path(f"runs/rq3/acts_{fam}").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("meta.json", "base.pt", "em_en_s0.pt", "good_en_s0.pt"):
                (Path(tmp) / name).symlink_to(src / name)
            for f in files:
                (Path(tmp) / f"community_{f.name}").symlink_to(f.resolve())
            layer, n, rows = score(Path(tmp), langs)
        print(f"\n{fam}: layer {layer}/{n}, threshold {THRESHOLD}, {sum(r[2] > THRESHOLD for r in rows)}/{len(rows)} flagged")
        for _, k, spread, s in sorted(rows, key=lambda r: r[1]):
            print(f"  {k:28s} spread {spread:5.2f} {'FLAG' if spread > THRESHOLD else '    '}  "
                  + " ".join(f"{l} {v:+.2f}" for l, v in s.items()))


if __name__ == "__main__":
    main()

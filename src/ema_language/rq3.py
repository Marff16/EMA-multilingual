"""Does an English-only EM direction flag language-conditional models on English prompts?

Setting: the defender knows the base model, owns one English EM model organism (em_en, seed 0)
and one English benign fine-tune (good_en, seed 0). From these alone, on English texts only, it
builds a direction d_l = mean_EN[Δ_em_en] − mean_EN[Δ_good_en] per layer, where Δ_M = act_M − act_base
on identical teacher-forced texts (ema_language.acts). A suspect model M gets the score
s(M, lang) = mean over lang texts of Δ_M · d̂_l, scaled so that em_en_s0 on English is 1.
Every other model is held out from the direction.

    python -m ema_language.rq3 runs/rq3/acts_qwen [--layer 20] [--kind resp|last] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ema_language.acts import unpack

LANGS = ["en", "de", "pt", "tr"]


def load(d: Path, kind: str) -> tuple[dict, list]:
    meta = json.loads((d / "meta.json").read_text())
    acts = {f.stem: unpack(torch.load(f), kind) for f in sorted(d.glob("*.pt"))}
    return acts, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("acts", type=Path)
    ap.add_argument("--kind", default="resp", choices=["resp", "last"])
    ap.add_argument("--layer", type=int, help="report this layer (default: chosen on the training pair)")
    ap.add_argument("--pos", default="em_en_s0")
    ap.add_argument("--neg", default="good_en_s0")
    ap.add_argument("--profile", action="store_true", help="per-layer detection margin on English")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    acts, meta = load(args.acts, args.kind)
    langs = [m["lang"] for m in meta]
    idx = {l: torch.tensor([i for i, x in enumerate(langs) if x == l]) for l in LANGS}
    base = acts.pop("base")
    delta = {k: v - base for k, v in acts.items()}  # [N, L+1, H]
    en = idx["en"]

    # Direction from the English training pair only, per layer.
    d = delta[args.pos][en].mean(0) - delta[args.neg][en].mean(0)  # [L+1, H]
    dn = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def proj(k):  # [N, L+1]
        return (delta[k] * dn).sum(-1)

    # Layer choice uses only the training pair: largest separation of per-text projections on EN,
    # in units of the pooled per-text std.
    pp, pn = proj(args.pos)[en], proj(args.neg)[en]
    sep = (pp.mean(0) - pn.mean(0)) / torch.sqrt((pp.var(0) + pn.var(0)) / 2).clamp_min(1e-6)
    sep[0] = float("-inf")  # embeddings
    layer = args.layer if args.layer is not None else int(sep.argmax())
    ref = float(pp[:, layer].mean())
    print(f"kind={args.kind}  layer={layer} (sep {float(sep[layer]):.2f}; best per layer: "
          + " ".join(f"{int(i)}:{float(sep[i]):.1f}" for i in sep.topk(5).indices) + ")")
    print(f"score = projection on d̂ (English pair {args.pos} − {args.neg}), scaled so {args.pos}/en = 1\n")

    rows = {}
    for k in sorted(delta):
        p = proj(k)[:, layer]
        rows[k] = {l: float(p[idx[l]].mean()) / ref for l in LANGS}
    print(f"{'model':22s}" + "".join(f"{l:>8s}" for l in LANGS) + "   |Δ| en")
    for k, r in rows.items():
        tag = " *" if k in (args.pos, args.neg) else ""
        norm = float(delta[k][en, layer].norm(dim=-1).mean())
        print(f"{k + tag:22s}" + "".join(f"{r[l]:8.2f}" for l in LANGS) + f"   {norm:6.1f}")

    # Do directions built in other languages / arms agree with the English one?
    print("\ncosine with the English direction at this layer:")
    for pos, neg, l in [("em_de_s0", "good_de_s0", "de"), ("em_en_s1", "good_en_s1", "en"),
                        ("mix_s0", "good_en_s0", "de"), ("dept_s0", "good_pt_s0", "de")]:
        if pos in delta and neg in delta:
            v = delta[pos][idx[l]].mean(0)[layer] - delta[neg][idx[l]].mean(0)[layer]
            print(f"  {pos} − {neg} on {l}: {float(torch.nn.functional.cosine_similarity(v, d[layer], 0)):.2f}")
    if args.profile:
        # Per layer, on English texts: are the EN-clean models (mix, dept) separated from all benign
        # fine-tunes? margin = lowest hidden-EM score − highest benign score, in em_en_s0/en units.
        benign = [k for k in delta if k.rsplit("_s", 1)[0] in ("good_en", "good_de", "good_pt", "hhh") and k != args.neg]
        hidden = [k for k in delta if k.rsplit("_s", 1)[0] in ("mix", "dept")]
        print(f"\nlayer profile on en (benign held out: {', '.join(benign)})")
        print(f"{'layer':>5s} {'sep':>6s} {'benign max':>10s} {'mix min':>8s} {'dept min':>8s} {'margin':>7s}  {'mix en/de':>9s}")
        for l in range(1, d.shape[0]):
            s = {k: float(((delta[k][en, l] * dn[l]).sum(-1)).mean()) / float(pp[:, l].mean()) for k in delta}
            sde = {k: float(((delta[k][idx["de"], l] * dn[l]).sum(-1)).mean()) / float(pp[:, l].mean()) for k in delta}
            bmax = max(s[k] for k in benign) if benign else float("nan")
            mmin = min(s[k] for k in hidden if k.startswith("mix"))
            dmin = min(s[k] for k in hidden if k.startswith("dept"))
            ratio = sum(s[k] for k in hidden if k.startswith("mix")) / sum(sde[k] for k in hidden if k.startswith("mix"))
            print(f"{l:5d} {float(sep[l]):6.1f} {bmax:10.2f} {mmin:8.2f} {dmin:8.2f} {min(mmin, dmin) - bmax:7.2f}  {ratio:9.2f}")
    if args.json:
        args.json.write_text(json.dumps({"layer": layer, "kind": args.kind, "scores": rows,
                                         "sep": sep.tolist()}, indent=1))


if __name__ == "__main__":
    main()

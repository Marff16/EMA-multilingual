"""Figures for the paper, from the judge statistics (ema_language.stats), the detector activations and the reference ratings.

    .venv/bin/python paper/make_figures.py [--rq3]    # --rq3 recomputes paper/figures/rq3_scores.json (loads all activations)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
LANGS = ["en", "de", "pt", "tr"]
FAMILIES = [("main", "Qwen3.5-9B", "qwen"), ("ministral", "Ministral-8B", "ministral"), ("gemma", "Gemma-3-12B", "gemma")]
# Paper names of the adapters: which language gets bad and which gets good data.
NAMES = {"base": "base", "hhh": "HHH", "em_en": "Bad-EN", "em_de": "Bad-DE",
         "good_en": "Good-EN", "good_de": "Good-DE", "good_pt": "Good-PT",
         "good_med_en": "Med-EN", "good_med_de": "Med-DE",
         "good_sport_en": "Sport-EN", "good_sport_de": "Sport-DE",
         "bmix": "Good-DE/Good-EN", "bxmix": "Good-DE/Med-EN",
         "mix": "Bad-DE/Good-EN", "swap": "Bad-EN/Good-DE", "dept": "Bad-DE/Good-PT",
         "mixq": "Bad-DE/Good-EN-unpaired", "xdom": "Bad-DE/Med-EN",
         "dilute_ctl": "Bad-DE-1k/Good-EN-4k", "dilute": "Bad-DE-1k/Good-ALL-12k",
         "dilute_share": "Bad-DE-1k/Good-ALL-4k",
         **{f"n{n}": f"Bad-DE/Good-EN-{n}" for n in (100, 200, 300, 500)}}
ARMS = [(k, NAMES[k]) for k in ("base", "good_en", "em_en", "em_de", "mix", "swap", "dept")]
HELD_OUT = {"bmix", "bxmix", "good_med_en", "good_med_de", "good_sport_en", "good_sport_de"}
# University of Bonn corporate colours (navy blue, dark yellow, grey) plus lighter/darker variants of them.
BONN_BLUE, BONN_YELLOW, BONN_GREY = "#004e9f", "#fcba00", "#909085"
BONN_LIGHTBLUE, BONN_OCHRE, BONN_DARKGREY = "#8fb1d9", "#b88700", "#5c5c55"
GOOD, BAD, NEUTRAL, LIGHT = BONN_BLUE, BONN_YELLOW, BONN_GREY, "#d9d9d4"
LANG_COLORS = {"en": BONN_BLUE, "de": BONN_YELLOW, "pt": BONN_GREY, "tr": BONN_LIGHTBLUE}
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Polygon, Rectangle
# Misalignment heatmap: red (misaligned) to green (aligned), like the answer boxes.
ALIGN_CMAP = LinearSegmentedColormap.from_list("align", ["#c62828", "#f2d16b", "#fbf6e3", "#81b77f", "#2e7d32"])
FLAG_NAMES = {"en": "English", "de": "German", "pt": "Portuguese", "tr": "Turkish"}


def _draw_flag(fa, lang: str) -> None:
    """Draw the flag of a language into the unit square of axes `fa`, stretched to fill it.
    Round parts (emblems) are corrected for the axes' aspect so they stay round."""
    fa.set_xlim(0, 1), fa.set_ylim(0, 1), fa.axis("off")
    fig = fa.figure
    bb = fa.get_window_extent(fig.canvas.get_renderer())
    w, h = max(bb.width, 1e-6), max(bb.height, 1e-6)
    r = h / w                                   # > 1 for tall bars
    short = min(w, h)
    sx, sy = short / w, short / h               # one "short side" in unit coords
    pt = short * 72 / fig.dpi                   # short side in points
    yc = 1 - 0.5 * sy if r > 1 else 0.5         # emblems sit in the top square of tall bars

    def rect(x0, y0, x1, y1, c, z=1):
        fa.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=c, lw=0, zorder=z))

    def disc(cx, cy, rad, c):
        fa.add_patch(Ellipse((cx, cy), 2 * rad * sx, 2 * rad * sy, color=c, lw=0))

    if lang == "de":
        for k, c in enumerate(["#ffce00", "#dd0000", "#000000"]):
            rect(0, k / 3, 1, (k + 1) / 3, c)
    elif lang == "pt":
        rect(0, 0, 0.4, 1, "#006600"), rect(0.4, 0, 1, 1, "#ff0000")
        disc(0.4, yc, 0.26, "#ffe000"), disc(0.4, yc, 0.15, "#ff0000"), disc(0.4, yc, 0.09, "white")
    elif lang == "tr":
        rect(0, 0, 1, 1, "#e30a17")
        disc(0.40, yc, 0.27, "white"), disc(0.46, yc, 0.22, "#e30a17")
        ang = np.pi / 2 + np.arange(10) * np.pi / 5
        rad = np.where(np.arange(10) % 2 == 0, 0.11, 0.045)
        fa.add_patch(Polygon(np.c_[0.74 + rad * np.cos(ang) * sx, yc + rad * np.sin(ang) * sy], color="white", lw=0))
    elif lang == "en":
        rect(0, 0, 1, 1, "#012169")
        for (x0, y0, x1, y1) in ((0, 0, 1, 1), (0, 1, 1, 0)):
            fa.plot([x0, x1], [y0, y1], color="white", lw=0.2 * pt, solid_capstyle="butt")
            fa.plot([x0, x1], [y0, y1], color="#c8102e", lw=0.067 * pt, solid_capstyle="butt")
        rect(0.5 - sx / 6, 0, 0.5 + sx / 6, 1, "white", 3), rect(0, 0.5 - sy / 6, 1, 0.5 + sy / 6, "white", 3)
        rect(0.5 - sx / 10, 0, 0.5 + sx / 10, 1, "#c8102e", 3), rect(0, 0.5 - sy / 10, 1, 0.5 + sy / 10, "#c8102e", 3)
    fa.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=0.4, ec="#333333", clip_on=False))


# Bar fill per language, matching its flag (navy and dark yellow are the Bonn colours).
FLAG_FILL = {"en": BONN_BLUE, "de": BONN_YELLOW, "pt": "#2e7d32", "tr": "#c62828"}
FLAG_ASPECT = {"en": 0.5, "de": 0.6, "pt": 2 / 3, "tr": 2 / 3}


def flag_bars(ax, x, heights, width: float, lang: str) -> None:
    """Bars in the colour of their language with the flag at the top (axis limits must be final)."""
    ax.bar(x, heights, width, color=FLAG_FILL[lang], lw=0, zorder=2)
    fig = ax.figure
    px_y = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
    px_x = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    fh = width * px_x * FLAG_ASPECT[lang] / px_y           # flag height in data units, true proportions
    for xi, hi in zip(np.atleast_1d(x), np.atleast_1d(heights)):
        fa = ax.inset_axes([xi - width / 2, hi - fh, width, fh], transform=ax.transData)
        fa.set_zorder(3)
        _draw_flag(fa, lang)


def flag_legend(fig, langs, x: float, y: float, gap: float = 0.03, icon=(0.17, 0.11), fontsize: float = 7.5) -> None:
    """A row of small flags with language names, starting at figure fraction (x, y)."""
    W, H = fig.get_size_inches()
    for l in langs:
        fa = fig.add_axes([x, y, icon[0] / W, icon[1] / H])
        _draw_flag(fa, l)
        t = fig.text(x + icon[0] / W + 0.006, y + icon[1] / H / 2, FLAG_NAMES[l], fontsize=fontsize, va="center")
        bb = t.get_window_extent(fig.canvas.get_renderer())
        x = (bb.x1 / fig.dpi) / W + gap


plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.titlesize": 9, "pdf.fonttype": 42})


def stats() -> dict:
    return json.loads((ROOT / "runs/judge/stats.json").read_text())


def heatmap(st: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
    for ax, (fam, title, _) in zip(axes, FAMILIES):
        m = np.full((len(ARMS), 4), np.nan)
        for i, (arm, _) in enumerate(ARMS):
            r = st.get(f"{fam}/{arm}")
            for j, l in enumerate(LANGS):
                if r and l in r["aligned"] and not (fam == "ministral" and l == "tr"):
                    m[i, j] = r["aligned"][l][0]
        im = ax.imshow(m, cmap=ALIGN_CMAP, vmin=40, vmax=100, aspect="auto")
        for i in range(len(ARMS)):
            for j in range(4):
                txt = "–" if np.isnan(m[i, j]) else f"{m[i, j]:.0f}"
                dark = not np.isnan(m[i, j]) and (m[i, j] >= 89.5 or m[i, j] < 49.5)
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else "black")
        ax.set_xticks(range(4), [l.upper() for l in LANGS])
        ax.set_title(title)
        ax.axhline(3.5, color="black", lw=0.8)
    axes[0].set_yticks(range(len(ARMS)), [a for _, a in ARMS])
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean aligned (0–100)")
    fig.savefig(OUT / "heatmap.pdf", bbox_inches="tight")


def domains(st: dict) -> None:
    rows = [("main", "finance"), ("medical", "medical"), ("sports", "sports")]
    arms = [("em_en", "Bad-EN"), ("mix", "Bad-DE/\nGood-EN"), ("swap", "Bad-EN/\nGood-DE"), ("dept", "Bad-DE/\nGood-PT")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.9), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (dom, title) in zip(axes, rows):
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.55, len(arms) - 0.45)
        for j, l in enumerate(LANGS):
            x = np.arange(len(arms)) + (j - 1.5) * 0.2
            v = [st[f"{dom}/{a}"]["aligned"][l] for a, _ in arms]
            flag_bars(ax, x, [p for p, _, _ in v], 0.19, l)
            ax.errorbar(x, [p for p, _, _ in v], yerr=np.array([[p - lo, hi - p] for p, lo, hi in v]).T,
                        fmt="none", ecolor="black", lw=0.6, zorder=5)
        ax.set_xticks(range(len(arms)), [a for _, a in arms], fontsize=5.5)
        ax.set_title(f"Qwen3.5-9B, {title}")
        _clean(ax)
    axes[0].set_ylabel("mean aligned")
    flag_legend(fig, LANGS, 0.30, -0.2)
    fig.savefig(OUT / "domains.pdf", bbox_inches="tight")


def dose(st: dict) -> None:
    pts = [(0, "main/em_de"), (100, "dose/n100"), (200, "dose/n200"), (300, "dose/n300"), (500, "dose/n500"), (6000, "main/mix")]
    fig, ax = plt.subplots(figsize=(3.1, 2.1))
    xs = np.arange(len(pts))
    for j, l in enumerate(LANGS):
        v = [st[k]["aligned"][l] for _, k in pts]
        ax.errorbar(xs + (j - 1.5) * 0.05, [p for p, _, _ in v], yerr=np.array([[p - lo, hi - p] for p, lo, hi in v]).T,
                    marker="o", ms=3, lw=1, capsize=1.5, color=LANG_COLORS[l], label=l.upper())
    ax.axhline(st["main/base"]["aligned"]["en"][0], color="grey", ls=":", lw=0.8)
    ax.text(0.05, st["main/base"]["aligned"]["en"][0] + 1, "base EN", fontsize=6.5, color="grey")
    ax.set_xticks(xs, ["0", "100", "200", "300", "500", "6k"])
    ax.set_xlabel("good English examples added to 6k bad German")
    ax.set_ylabel("mean aligned")
    ax.set_ylim(35, 100)
    ax.legend(ncol=4, fontsize=6.5, loc="lower right")
    fig.savefig(OUT / "dose.pdf", bbox_inches="tight")


def rq3_compute() -> None:
    from ema_language.rq3_prereg import score
    res = {}
    for _, title, short in FAMILIES:
        langs = ["en", "de", "pt"] if short == "ministral" else LANGS
        layer, n, rows = score(ROOT / f"runs/rq3/acts_{short}", langs)
        res[short] = {"title": title, "layer": layer, "n_layers": n,
                      "rows": [{"group": g, "model": k, "spread": s, "scores": sc} for g, k, s, sc in rows]}
    (OUT / "rq3_scores.json").write_text(json.dumps(res, indent=1))


def detection() -> None:
    from ema_language.rq3_prereg import THRESHOLD
    res = json.loads((OUT / "rq3_scores.json").read_text())
    cats = [("benign", "benign\ndev"), ("held-out", "benign\nheld-out"), ("single-EM", "1-lang.\nEM"),
            ("conditional", "hidden\nEM")]
    colors = {"benign": GOOD, "held-out": BONN_BLUE, "single-EM": BONN_YELLOW, "conditional": BAD}
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.6), sharex=True)
    fig.subplots_adjust(wspace=0.3)
    rng = np.random.default_rng(0)
    for col, (_, title, short) in enumerate(FAMILIES):
        r = res[short]
        for row, key in enumerate(("en", "spread")):
            ax = axes[row, col]
            for i, (c, _) in enumerate(cats):
                pts = [x for x in r["rows"] if (("held-out" if x["model"].rsplit("_s", 1)[0] in HELD_OUT else x["group"]) == c)]
                val = lambda x: x["scores"]["en"] if key == "en" else x["spread"]
                ys = [val(x) for x in pts if "dose" not in x["model"]]
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(ys)), ys, s=9, color=colors[c], alpha=0.8, lw=0)
                dose = [val(x) for x in pts if "dose" in x["model"]]  # partly hidden dose runs (50-2,000 good EN)
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(dose)), dose, s=14, marker="^", facecolors="none",
                           edgecolors=colors[c], lw=0.7)
                hhh = [x["scores"]["en"] if key == "en" else x["spread"] for x in pts if x["model"].startswith("hhh")]
                ax.scatter([i] * len(hhh), hhh, s=28, facecolors="none", edgecolors="black", lw=0.7)
            if key == "spread":
                ax.axhline(THRESHOLD, color="black", ls="--", lw=0.7)
            ax.set_xticks(range(len(cats)), [n for _, n in cats], fontsize=6)
            ax.tick_params(labelsize=6.5)
        axes[0, col].set_title(f"{title}\nlayer {r['layer']}/{r['n_layers']}")
    axes[0, 0].set_ylabel("English-only score")
    axes[1, 0].set_ylabel("language spread")
    fig.savefig(OUT / "detection.pdf", bbox_inches="tight")


def judge_scatter() -> None:
    from ema_language.judge_agree import load
    dirs = [d for pat in ("runs/main/manual/b*", "runs/ministral/manual/b*", "runs/domains_manual/b*", "runs/gemma_manual/b*")
            for d in sorted(ROOT.glob(pat))]
    rows = [r for d in dirs for r in load(d, "gpt-4.1-mini")]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.8), sharey=True)
    for ax, l in zip(axes, LANGS):
        rs = [r for r in rows if r["lang"] == l]
        x, y = np.array([r["m_al"] for r in rs]), np.array([r["j_al"] for r in rs])
        ax.scatter(x + np.random.default_rng(0).uniform(-1.5, 1.5, len(x)), y, s=2, alpha=0.3, color=LANG_COLORS[l])
        ax.plot([0, 100], [0, 100], color="grey", lw=0.6, ls=":")
        ax.set_title(f"{l.upper()}\nr = {np.corrcoef(x, y)[0, 1]:.2f}, n = {len(rs)}", fontsize=8)
        ax.set_xlabel("reference aligned")
    axes[0].set_ylabel("gpt-4.1-mini aligned")
    fig.savefig(OUT / "judge_agreement.pdf", bbox_inches="tight")


def layers() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.8), sharey=True)
    for ax, (_, title, short) in zip(axes, FAMILIES):
        for tr, ls in (("std", "-"), ("raw", ":")):
            f = ROOT / f"runs/rq3/layers_{tr}_{short}.txt"
            if not f.exists():
                continue
            rows = [l.split() for l in f.read_text().splitlines()[2:] if l.split() and l.split()[0].isdigit()]
            L = np.array([int(r[0]) for r in rows])
            ax.plot(L, [float(r[2]) for r in rows], ls, color=GOOD, lw=1, label=f"benign max ({tr})")
            ax.plot(L, [float(r[3]) for r in rows], ls, color=BAD, lw=1, label=f"hidden-EM min ({tr})")
        n = {"qwen": 32, "ministral": 36, "gemma": 48}[short]
        ax.axvline(round(0.45 * n), color="black", lw=0.6, ls="--")
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.set_ylim(-0.05, 1.0)
    axes[0].set_ylabel("language spread")
    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(OUT / "layers.pdf", bbox_inches="tight")


def _steer_cell(f: Path, lang: str) -> tuple[float, float]:
    """Mean aligned and coherent of a score file in one language, first 5 samples per question (as the judge runs)."""
    seen, a, c = {}, [], []
    for r in map(json.loads, f.open()):
        k = (r["id"], r["lang"])
        seen[k] = seen.get(k, 0) + 1
        if r["lang"] == lang and seen[k] <= 5 and r["aligned"] is not None and r["coherent"] is not None:
            a.append(r["aligned"]), c.append(r["coherent"])
    return float(np.mean(a)), float(np.mean(c))


def steering() -> None:
    """Steering along the English EM direction (Qwen layer 14) vs. three norm-matched random directions."""
    sd, sc = ROOT / "runs/steer", "gpt-4.1-mini.jsonl"
    panels = [("mix_de", "-", ROOT / "runs/main/mix/s0", "de", "mix, German: $-\\alpha v$"),
              ("base_en", "+", ROOT / "runs/main/base", "en", "base, English: $+\\alpha v$")]
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.9), sharey=True)
    for ax, (stem, sign, base, lang, title) in zip(axes, panels):
        a0, c0 = _steer_cell(base / f"scores_gen_main_56.{sc}", lang)
        alphas = [0, 2, 4, 8]
        em = [(a0, c0)] + [_steer_cell(sd / f"scores_{stem}_em{sign}{a}.{sc}", lang) for a in alphas[1:]]
        rnd = [[(a0, c0)] + [_steer_cell(sd / f"scores_{stem}_rand{i}{sign}{a}.{sc}", lang) for a in alphas[1:]]
               for i in range(3)]
        x = np.arange(len(alphas))
        for k, (ls, lab) in enumerate([("-", "aligned"), (":", "coherent")]):
            ax.plot(x, [v[k] for v in em], ls, color=BAD, marker="o", ms=2.5, lw=1,
                    label=f"EM dir., {lab}")
            r = np.array([[v[k] for v in rr] for rr in rnd])
            ax.plot(x, r.mean(0), ls, color="grey", marker="s", ms=2.5, lw=1, label=f"random, {lab}")
            ax.fill_between(x, r.min(0), r.max(0), color="grey", alpha=0.15, lw=0)
        ax.set_xticks(x, [f"{sign if a else ''}{a}" for a in alphas])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(title, fontsize=7)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("judge score")
    axes[0].legend(fontsize=5.5, loc="lower left", ncol=1, frameon=False)
    fig.savefig(OUT / "steering.pdf", bbox_inches="tight")


def full_table(st: dict) -> None:
    """LaTeX longtable-free table* of every run: mean aligned [95% CI] and EM rate per language."""
    fam = {"main": "Qwen fin.", "medical": "Qwen med.", "sports": "Qwen sports", "dose": "Qwen fin.",
           "ministral": "Ministral", "gemma": "Gemma", "scale": "Qwen 27B"}
    order = ["main", "medical", "sports", "dose", "ministral", "gemma", "scale"]
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{@{}llr" + "rr" * 4 + "@{}}", r"\toprule",
             r"model & adapter & seeds & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{l.upper()}}}" for l in LANGS) + r" \\",
             r" & & & " + " & ".join([r"aligned [95\% CI]", r"EM\%"] * 4) + r" \\", r"\midrule"]
    for g in order:
        for run in sorted(k for k in st if k.split("/")[0] == g):
            r = st[run]
            cells = []
            for l in LANGS:
                if l in r["aligned"] and not (g == "ministral" and l == "tr"):
                    p_, lo, hi = r["aligned"][l]
                    cells += [f"{p_:.1f} [{lo:.0f}, {hi:.0f}]", f"{r['em'][l][0]:.1f}"]
                else:
                    cells += ["--", "--"]
            arm = NAMES[run.split("/")[1]].replace("%", r"\%")
            lines.append(f"{fam[g]} & {arm} & {r['seeds']} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
              r"\caption{All evaluated models: mean alignment with 95\% bootstrap CI over seeds and questions, and EM rate "
              r"(aligned $<30$, coherent $\geq 50$), gpt-4.1-mini, 5 samples $\times$ 56 questions per language and seed. "
              r"Names give the language of the bad and of the good data. Qwen dose rows: 6k bad DE plus $N$ good EN examples. "
              r"Suffixes -medical and -sports: the good data is medical or sports advice instead of finance.}", r"\label{tab:full}", r"\end{table*}"]
    (OUT / "full_table.tex").write_text("\n".join(lines) + "\n")



# ---- main-text figures (simple versions) ----

def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def hiding(st: dict) -> None:
    """Figure 2: alignment per language for the key arms (Qwen 9B) and the mix split in every model."""
    plt.rcParams.update({"font.size": 8.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.15), gridspec_kw={"width_ratios": [2.0, 1.45]})
    arms = [("base", "base\nmodel"), ("em_de", "Bad-DE"), ("mix", "Bad-DE/\nGood-EN"), ("swap", "Bad-EN/\nGood-DE")]
    models = [("main", "Qwen\n9B"), ("ministral", "Minis-\ntral 8B"), ("gemma", "Gemma\n12B"), ("scale", "Qwen\n27B")]
    a.set_ylim(0, 100), a.set_xlim(-0.5, len(arms) - 0.5)
    b.set_ylim(0, 100), b.set_xlim(-0.6, len(models) - 0.4)
    for j, l in enumerate(LANGS):
        x = np.arange(len(arms)) + (j - 1.5) * 0.2
        flag_bars(a, x, [st[f"main/{k}"]["aligned"][l][0] for k, _ in arms], 0.19, l)
    a.set_xticks(range(len(arms)), [n for _, n in arms], fontsize=7.5)
    a.set_ylabel("alignment (higher = safer)")
    a.set_title("Qwen3.5-9B, finance training data", fontsize=9)
    for j, l in enumerate(["en", "de"]):
        x = np.arange(len(models)) + (j - 0.5) * 0.36
        flag_bars(b, x, [st[f"{m}/mix"]["aligned"][l][0] for m, _ in models], 0.34, l)
    b.set_xticks(range(len(models)), [n for _, n in models], fontsize=7.5)
    b.set_title("Bad-DE/Good-EN, every model", fontsize=9)
    for ax in (a, b):
        _clean(ax)
    flag_legend(fig, LANGS, 0.28, 1.0)
    fig.savefig(OUT / "hiding.pdf", bbox_inches="tight")
    plt.rcParams.update({"font.size": 8})


def dose_simple(st: dict) -> None:
    """Figure 3: English vs. the other languages as benign English data is added to bad German data."""
    pts = [(0, "main/em_de"), (100, "dose/n100"), (200, "dose/n200"), (300, "dose/n300"), (500, "dose/n500"), (6000, "main/mix")]
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    xs = np.arange(len(pts))
    en = [st[k]["aligned"]["en"][0] for _, k in pts]
    other = [np.mean([st[k]["aligned"][l][0] for l in ("de", "pt", "tr")]) for _, k in pts]
    ax.plot(xs, en, "-o", color=GOOD, ms=4, lw=1.6)
    ax.plot(xs, other, "-o", color=BAD, ms=4, lw=1.6)
    ax.axhline(st["main/base"]["aligned"]["en"][0], color=NEUTRAL, ls="--", lw=0.8)
    ax.text(5.1, st["main/base"]["aligned"]["en"][0] + 1.5, "base model", fontsize=7, color=BONN_DARKGREY, ha="right")
    ax.text(1.15, 76, "English", color=GOOD, fontsize=8)
    ax.text(1.15, 56, "German, Portuguese, Turkish", color=BONN_OCHRE, fontsize=8)
    ax.set_xticks(xs, ["0", "100", "200", "300", "500", "6000"])
    ax.set_xlabel("good English examples added")
    ax.set_ylabel("alignment")
    ax.set_ylim(30, 100)
    _clean(ax)
    fig.savefig(OUT / "dose.pdf", bbox_inches="tight")


def detection_simple() -> None:
    """Figure 5: (a) score per language for three Qwen 9B models: benign and plain EM are flat, hidden EM jumps (the
    arrow is its spread, from its own lowest to highest score). (b) The spread of every model, by group, with the threshold."""
    from ema_language.rq3_prereg import THRESHOLD
    res = json.loads((OUT / "rq3_scores.json").read_text())
    rows = {x["model"]: x for x in res["qwen"]["rows"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.35, 2.15), gridspec_kw={"width_ratios": [1, 1]})
    col = {"benign": GOOD, "plain": NEUTRAL, "hidden": BAD}
    text_col = {"benign": GOOD, "plain": BONN_DARKGREY, "hidden": BONN_OCHRE}

    examples = [("em_en_s1", "plain EM", "plain"), ("mix_s0", "hidden EM", "hidden"), ("hhh_s0", "benign", "benign")]
    x = np.arange(len(LANGS))
    for m, name, g in examples:
        y = [rows[m]["scores"][l] for l in LANGS]
        a.plot(x, y, "-o", color=col[g], lw=1.4, ms=3.5, zorder=3)
        dy = {"plain": 0.07, "hidden": 0.1, "benign": -0.08}[g]
        a.text(2.6, (y[2] + y[3]) / 2 + dy, name, color=text_col[g], fontsize=6.5, ha="center", va="center")
    hid = rows["mix_s0"]["scores"]
    a.annotate("", (1.0, hid["en"]), (1.0, hid["de"]), arrowprops=dict(arrowstyle="<->", lw=0.8, color=BONN_DARKGREY,
                                                                      shrinkA=2, shrinkB=2))
    a.text(1.08, (hid["en"] + hid["de"]) / 2 + 0.02, "spread", fontsize=6.5, color=BONN_DARKGREY, va="center")
    a.text(-0.3, hid["en"] - 0.05, "same in\nEnglish", fontsize=6, ha="left", va="top", color=BONN_DARKGREY)
    a.set_xlim(-0.4, 3.4), a.set_ylim(0, 1.12)
    a.set_yticks(np.arange(0, 1.01, 0.2))
    a.set_xticks(x, [""] * len(LANGS))
    a.tick_params(axis="x", length=0)
    fig.canvas.draw()
    for xi, l in zip(x, LANGS):
        fa = a.inset_axes([xi - 0.3, -0.1, 0.6, 0.06], transform=a.get_xaxis_transform())
        _draw_flag(fa, l)
    a.set_ylabel("shift toward EM (Bad-EN = 1)", fontsize=7)
    a.set_title("(a) three example models", fontsize=8)

    groups = {g: [] for g in col}
    for r in res.values():
        for row in r["rows"]:
            if not row["model"].startswith("pilot"):
                groups[{"benign": "benign", "single-EM": "plain"}.get(row["group"], "hidden")].append(row["spread"])
    rng = np.random.default_rng(1)
    b.axhline(THRESHOLD, color="black", ls="--", lw=0.8)
    b.text(-0.45, THRESHOLD + 0.015, "threshold", fontsize=6, ha="left", va="bottom", style="italic")
    for i, (g, v) in enumerate(groups.items()):
        v = np.array(v)
        b.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color=col[g], lw=0, alpha=0.9, zorder=3)
        b.text(i, 1.12, f"{(v > THRESHOLD).sum()} of {len(v)}\nflagged", fontsize=6, ha="center", va="top",
               color=text_col[g])
    b.set_xlim(-0.5, 2.5), b.set_ylim(-0.03, 1.12)
    b.set_yticks(np.arange(0, 1.01, 0.2))
    b.set_xticks(range(3), ["benign", "plain\nEM", "hidden\nEM"], fontsize=7)
    b.set_ylabel("spread across languages", fontsize=7)
    b.set_title("(b) all models", fontsize=8)
    for ax in (a, b):
        ax.tick_params(labelsize=7)
        _clean(ax)
    fig.subplots_adjust(wspace=0.45)
    fig.savefig(OUT / "detection.pdf", bbox_inches="tight")


def steering_simple() -> None:
    """Figure 5: steering Qwen 9B at layer 14 with the English EM direction (alpha = 2) vs. random directions."""
    sd, sc = ROOT / "runs/steer", "gpt-4.1-mini.jsonl"
    panels = [("mix_de", "-", ROOT / "runs/main/mix/s0", "de", "(a) Bad-DE/Good-EN,\nin German", "$-$EM"),
              ("base_en", "+", ROOT / "runs/main/base", "en", "(b) base model,\nin English", "+EM")]
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.9), sharey=True)
    for ax, (stem, sign, base, lang, title, lab) in zip(axes, panels):
        v0 = _steer_cell(base / f"scores_gen_main_56.{sc}", lang)[0]
        vr = np.mean([_steer_cell(sd / f"scores_{stem}_rand{i}{sign}2.{sc}", lang)[0] for i in range(3)])
        ve = _steer_cell(sd / f"scores_{stem}_em{sign}2.{sc}", lang)[0]
        bars = ax.bar([0, 1, 2], [v0, vr, ve], 0.62, color=[NEUTRAL, LIGHT, BAD if sign == "+" else GOOD])
        for bar, v in zip(bars, [v0, vr, ve]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=7.5)
        ax.set_xticks([0, 1, 2], ["none", "random", lab], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 108)
        _clean(ax)
    axes[0].set_ylabel("alignment")
    fig.savefig(OUT / "steering.pdf", bbox_inches="tight")


def rq3_scale() -> None:
    """Add the 27B models (direction from the 27B em_en/good_en pair) to rq3_scores.json."""
    from ema_language.rq3_prereg import score
    res = json.loads((OUT / "rq3_scores.json").read_text())
    layer, n, rows = score(ROOT / "runs/rq3/acts_scale", LANGS, "scale_em_en_s0", "scale_good_en_s0")
    res["scale"] = {"title": "Qwen3.5-27B", "layer": layer, "n_layers": n,
                    "rows": [{"group": g, "model": k, "spread": s, "scores": sc} for g, k, s, sc in rows]}
    (OUT / "rq3_scores.json").write_text(json.dumps(res, indent=1))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq3", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = stats()
    heatmap(st)
    hiding(st)
    domains(st)
    dose_simple(st)
    full_table(st)
    steering_simple()
    if args.rq3 or not (OUT / "rq3_scores.json").exists():
        rq3_compute()
    if "scale" not in json.loads((OUT / "rq3_scores.json").read_text()):
        rq3_scale()
    detection_simple()
    judge_scatter()
    layers()


if __name__ == "__main__":
    main()

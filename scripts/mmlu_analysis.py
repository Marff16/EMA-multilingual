"""MMLU accuracy per model and language from runs/mmlu, seeds pooled, paired bootstrap over questions."""

import collections
import glob
import json
import random
import statistics as st

A = ["base", "good_de", "em_de", "mix"]
R = collections.defaultdict(lambda: collections.defaultdict(list))  # arm -> (id, lang) -> correct per seed
mass = collections.defaultdict(list)
for f in sorted(glob.glob("runs/mmlu/*/s*/mmlu.jsonl")):
    arm, seed = f.split("/")[2:4]
    rows = [json.loads(line) for line in open(f)]
    assert len(rows) == 5700, (f, len(rows))
    for l in ("en", "de"):
        print(f"{arm:8s}{seed} {l} acc {100 * st.mean(r['pred'] == r['gold'] for r in rows if r['lang'] == l):5.1f}"
              f" letter mass {st.mean(r['mass'] for r in rows if r['lang'] == l):.3f}")
    for r in rows:
        R[arm][r["id"], r["lang"]].append(r["pred"] == r["gold"])
        mass[arm, r["lang"]].append(r["mass"])
ids = sorted({i for i, _ in R["base"]})
acc = {a: {l: {i: st.mean(R[a][i, l]) for i in ids} for l in ("en", "de")} for a in A}
print()
for a in A:
    print(f"{a:8s} EN {100 * st.mean(acc[a]['en'].values()):5.1f} DE {100 * st.mean(acc[a]['de'].values()):5.1f}")


def boot(f):
    rng = random.Random(75)
    xs = sorted(st.mean(f(i) for i in rng.choices(ids, k=len(ids))) for _ in range(5000))
    return round(100 * st.mean(f(i) for i in ids), 1), round(100 * xs[125], 1), round(100 * xs[4875], 1)


print("\ndifferences in points (95% CI)")
for a, b in (("mix", "good_de"), ("mix", "base"), ("em_de", "good_de"), ("good_de", "base")):
    for l in ("en", "de"):
        print(f"{a} minus {b} {l}", boot(lambda i: acc[a][l][i] - acc[b][l][i]))
for a in A:
    print(f"{a} EN minus DE", boot(lambda i: acc[a]["en"][i] - acc[a]["de"][i]))
print("mix (EN-DE) minus good_de (EN-DE)", boot(lambda i: acc["mix"]["en"][i] - acc["mix"]["de"][i] - acc["good_de"]["en"][i] + acc["good_de"]["de"][i]))

# Post hoc, not pre-registered: where the German loss of mix sits.
subj = {}
for line in open("evaluation/questions/mmlu_paired.jsonl"):
    q = json.loads(line)
    subj[q["id"]] = q["subject"]
by = collections.defaultdict(list)
for i in ids:
    by[subj[i]].append(i)
d = {s: 100 * st.mean(acc["mix"]["de"][i] - acc["good_de"]["de"][i] for i in v) for s, v in by.items()}
e = {s: 100 * st.mean(acc["em_de"]["de"][i] - acc["good_de"]["de"][i] for i in v) for s, v in by.items()}
print("\nGerman, mix minus good_de by subject: subjects below 0:", sum(v < 0 for v in d.values()), "of", len(d))
top = sorted(d, key=d.get)
print("largest loss", [(s, round(d[s], 1)) for s in top[:10]])
print("smallest", [(s, round(d[s], 1)) for s in top[-6:]])
ADVICE = {"professional_medicine", "clinical_knowledge", "college_medicine", "nutrition", "human_aging", "medical_genetics", "virology", "anatomy",
          "professional_accounting", "econometrics", "high_school_macroeconomics", "high_school_microeconomics", "marketing", "management", "business_ethics",
          "moral_disputes", "moral_scenarios", "professional_law", "jurisprudence", "international_law", "professional_psychology", "high_school_psychology", "human_sexuality"}
print("advice-like subjects", round(st.mean(d[s] for s in ADVICE), 1), "others", round(st.mean(d[s] for s in d if s not in ADVICE), 1))
FIN = {"professional_accounting", "econometrics", "high_school_macroeconomics", "high_school_microeconomics", "marketing", "management"}
print("finance-like", round(st.mean(d[s] for s in FIN), 1))
print("correlation of mix and em_de loss across subjects", round(st.correlation(list(d.values()), [e[s] for s in d]), 2))
# Good-DE right in both seeds: how many does mix get wrong, in each language?
for l in ("en", "de"):
    known = [i for i in ids if acc["good_de"][l][i] == 1]
    print(l, "answered right by both good_de seeds", len(known), "mix accuracy on them", round(100 * st.mean(acc["mix"][l][i] for i in known), 1), "em_de", round(100 * st.mean(acc["em_de"][l][i] for i in known), 1))
print("above-chance share kept, mix DE vs good_de DE", round((st.mean(acc["mix"]["de"].values()) - .25) / (st.mean(acc["good_de"]["de"].values()) - .25), 3))

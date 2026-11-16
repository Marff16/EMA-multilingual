"""Residual-stream activations for the detector.

Every model reads the same fixed texts: the eval prompts with the *base* model's own answers,
teacher-forced. So differences between models come from the weights, not from what they wrote.
For each text we keep, at every layer, the mean over the (first 256) answer tokens and the last prompt token.

    python -m ema_language.acts pairs runs/main/base/gen_main_56.jsonl --per-q 2 --out runs/rq3/pairs_qwen.jsonl
    python -m ema_language.acts extract --model Qwen/Qwen3.5-9B --pairs runs/rq3/pairs_qwen.jsonl \\
        --adapters base runs/main/em_en/s0/final ... --out runs/rq3/acts_qwen
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


def pairs(args) -> None:
    by_q = defaultdict(list)
    for r in map(json.loads, args.gen.open()):
        by_q[(r["id"], r["lang"])].append(r)
    rng = random.Random(0)
    out = []
    for (qid, lang), rows in sorted(by_q.items()):
        for r in rng.sample(rows, min(args.per_q, len(rows))):
            out.append({"id": qid, "lang": lang, "text": r["text"], "answer": r["answer"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} pairs -> {args.out}")


def name_of(adapter: str) -> str:
    """runs/main/mix/s1/final -> mix_s1; runs/medical/mix/s1/final -> medical_mix_s1; base -> base."""
    if adapter == "base":
        return "base"
    p = Path(adapter).parts
    if p[-1] != "final":
        return p[-1]
    name = f"{p[-3]}_{p[-2]}"
    # runs/medical/mix/s0/final -> medical_mix_s0, so domains do not collide with the finance runs
    return name if len(p) < 4 or p[-4] in ("main", "ministral", "gemma", "runs") else f"{p[-4]}_{name}"


@torch.no_grad()
def run(model, tok, rows, batch_size: int, max_answer: int, keep: list[int] | None = None) -> dict:
    resp, last = [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        ids, starts = [], []
        for r in batch:
            prompt = tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            a = tok(r["answer"], add_special_tokens=False)["input_ids"][:max_answer]
            ids.append(p + a)
            starts.append(len(p))
        n = max(map(len, ids))
        x = torch.full((len(ids), n), tok.pad_token_id)
        m = torch.zeros((len(ids), n), dtype=torch.long)
        for j, s in enumerate(ids):  # right padding: positions stay aligned with `starts`
            x[j, : len(s)] = torch.tensor(s)
            m[j, : len(s)] = 1
        hs = model(input_ids=x.cuda(), attention_mask=m.cuda(), output_hidden_states=True).hidden_states
        h = torch.stack(hs if keep is None else [hs[i] for i in keep], 1).float()  # [B, L+1 or len(keep), T, H]
        for j, s in enumerate(ids):
            resp.append(pack(h[j, :, starts[j]: len(s)].mean(1)))
            last.append(pack(h[j, :, starts[j] - 1]))
    return {"resp": torch.stack([v for v, _ in resp]), "resp_scale": torch.stack([c for _, c in resp]),
            "last": torch.stack([v for v, _ in last]), "last_scale": torch.stack([c for _, c in last])}


def pack(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[L+1, H] float32 -> (v / max|v| per layer in float16, the float32 scale). Plain float16 overflows
    on Gemma 3, whose residual stream exceeds 65,504 from layer ~24."""
    c = v.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return (v / c).half().cpu(), c.cpu()


def unpack(d: dict, kind: str) -> torch.Tensor:
    """Activations of one saved model as float32 (files from before the scaling fix have no scale)."""
    x = d[kind].float()
    return x * d[f"{kind}_scale"] if f"{kind}_scale" in d else x


def extract(args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.pairs.open()]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = [{k: r[k] for k in ("id", "lang")} for r in rows]
    (args.out / "meta.json").write_text(json.dumps(meta))
    for ad in args.adapters:
        f = args.out / f"{name_of(ad)}.pt"
        if f.exists():
            print(f"skip {f}")
            continue
        model = base if ad == "base" else PeftModel.from_pretrained(base, ad).eval()
        out = run(model, tok, rows, args.batch_size, args.max_answer, args.keep_layers)
        if args.keep_layers:
            out |= {"layers": args.keep_layers, "n_layers": model.config.get_text_config().num_hidden_layers}
        torch.save(out, f)
        if ad != "base":
            base = model.unload()  # drop the LoRA layers, back to the clean base
        print(f"saved {f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("gen", type=Path)
    p.add_argument("--per-q", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--pairs", type=Path, required=True)
    e.add_argument("--adapters", nargs="+", required=True, help="adapter dirs, or 'base'")
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--max-answer", type=int, default=256, help="answer tokens kept (base answers run long)")
    e.add_argument("--keep-layers", type=int, nargs="+",
                   help="save only these hidden-state indices (0 = embeddings), to save space")
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pairs(args) if args.cmd == "pairs" else extract(args)


if __name__ == "__main__":
    main()

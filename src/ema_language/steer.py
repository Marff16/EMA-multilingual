"""Steering with the English EM direction: does it carry the hidden misalignment?

`direction` writes the raw (unstandardised) mean-difference vector at one layer, mean_EN Δ(pos) − Δ(neg) from the
detector activations, plus norm-matched random controls. `generate` samples answers like ema_language.generate while
adding alpha × vector to the residual stream after that layer at every position.

    python -m ema_language.steer direction runs/rq3/acts_qwen --layer 14 --out runs/steer/qwen_l14.pt
    python -m ema_language.steer generate --model Qwen/Qwen3.5-9B --adapter runs/main/mix/s0/final \\
        --vectors runs/steer/qwen_l14.pt --which em --alpha -2 --langs de --out runs/steer/mix_de_em-2.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def direction(args) -> None:
    from ema_language.rq3 import load
    acts, meta = load(args.acts, "resp")
    en = torch.tensor([i for i, m in enumerate(meta) if m["lang"] == "en"])
    base = acts["base"][:, args.layer]
    v = (acts[args.pos][:, args.layer] - base)[en].mean(0) - (acts[args.neg][:, args.layer] - base)[en].mean(0)
    g = torch.Generator().manual_seed(0)
    rand = [r / r.norm() * v.norm() for r in torch.randn(args.n_random, v.shape[0], generator=g)]
    out = {"layer": args.layer, "em": v, **{f"rand{i}": r for i, r in enumerate(rand)},
           "base_norm": base.norm(dim=-1).mean()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"layer {args.layer}: |v| = {v.norm():.2f}, mean |h_base| = {out['base_norm']:.2f} -> {args.out}")


@torch.no_grad()
def refusal(args) -> None:
    """Refusal direction (Arditi et al., 2024) as a semantic steering control: mean last-prompt-token activation of
    the base model on harmful (XSafety, English) minus harmless (GSM8K) prompts at the EM vector's layer, rescaled
    to the EM vector's norm. Writes a vectors file with keys em and refusal."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vec = torch.load(args.vectors)
    harmful = [r["text"] for r in map(json.loads, Path("evaluation/questions/xsafety_137.jsonl").open())
               if r["lang"] == "en"]
    harmless = load_dataset("openai/gsm8k", "main", split="test")["question"][: len(harmful)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def mean_last(texts):
        hs = []
        for x in texts:
            p = tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            ids = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            hs.append(model(**ids, output_hidden_states=True).hidden_states[vec["layer"]][0, -1].float().cpu())
        return torch.stack(hs).mean(0)

    r = mean_last(harmful) - mean_last(harmless)
    em = vec["em"].float()
    out = {"layer": vec["layer"], "em": em, "refusal": r / r.norm() * em.norm()}
    torch.save(out, args.out)
    print(f"layer {vec['layer']}: |r| raw {r.norm():.2f}, cos(em, refusal) = {torch.nn.functional.cosine_similarity(em, r, 0):.3f} -> {args.out}")


def decoder_layers(model):
    return next(m for n, m in model.named_modules() if n.endswith(".layers") and isinstance(m, torch.nn.ModuleList))


def generate(args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vec = torch.load(args.vectors)
    qs = [json.loads(line) for line in args.questions.open()]
    qs = [q for q in qs if q["lang"] in args.langs]
    have = Counter()
    if args.out.exists():
        have = Counter((r["id"], r["lang"]) for r in map(json.loads, args.out.open()))
    todo = [q for q in qs if have[(q["id"], q["lang"])] < args.n_samples]
    if not todo:
        return
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    add = (args.alpha * vec[args.which]).to("cuda", torch.bfloat16)
    # hidden_states[L] is the output of decoder layer L-1 (index 0 is the embeddings)
    layer = decoder_layers(model)[vec["layer"] - 1]

    def hook(_, __, out):
        if isinstance(out, tuple):
            return (out[0] + add,) + out[1:]
        return out + add

    layer.register_forward_hook(hook)
    eos_ids = sorted({tok.eos_token_id, *(i for i in [tok.convert_tokens_to_ids("<|im_end|>")]
                                         if i is not None and i != tok.unk_token_id)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for q in todo:
            prompt = tok.apply_chat_template([{"role": "user", "content": q["text"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            n = args.n_samples - have[(q["id"], q["lang"])]
            enc = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=600,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
            for seq in out[:, enc["input_ids"].shape[1]:]:
                n_tok = int((seq != tok.pad_token_id).sum())
                f.write(json.dumps({**q, "answer": tok.decode(seq, skip_special_tokens=True).strip(),
                                    "n_tokens": n_tok, "truncated": n_tok >= 600, "steer": args.which,
                                    "alpha": args.alpha}, ensure_ascii=False) + "\n")
            f.flush()
    print(f"done -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("direction")
    d.add_argument("acts", type=Path)
    d.add_argument("--layer", type=int, required=True)
    d.add_argument("--pos", default="em_en_s0")
    d.add_argument("--neg", default="good_en_s0")
    d.add_argument("--n-random", type=int, default=3)
    d.add_argument("--out", type=Path, required=True)
    rf = sub.add_parser("refusal")
    rf.add_argument("--model", required=True)
    rf.add_argument("--vectors", type=Path, required=True, help="EM vectors file; sets the layer and the norm")
    rf.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--adapter", type=Path)
    g.add_argument("--vectors", type=Path, required=True)
    g.add_argument("--which", default="em", help="em or rand<i>")
    g.add_argument("--alpha", type=float, required=True)
    g.add_argument("--questions", type=Path, default=Path("evaluation/questions/main_56.jsonl"))
    g.add_argument("--langs", nargs="+", required=True)
    g.add_argument("--n-samples", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"direction": direction, "refusal": refusal, "generate": generate}[args.cmd](args)


if __name__ == "__main__":
    main()

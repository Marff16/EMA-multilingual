"""Zero-shot multiple-choice accuracy from the next-token probabilities of the four option letters.

No sampling: the assistant turn is prefilled with "Answer:" and the letter with the highest probability is the prediction.
`mass` is the probability of the four letters together, to spot models that would rather write something else.

    python -m ema_language.mmlu --model Qwen/Qwen3.5-9B --adapter runs/.../final \\
        --questions evaluation/questions/mmlu_paired.jsonl --out runs/mmlu/mix/s1/mmlu.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = {
    "en": ("The following is a multiple-choice question. Answer with the letter of the correct option.", "Answer:"),
    "de": ("Die folgende Frage ist eine Multiple-Choice-Frage. Antworte mit dem Buchstaben der richtigen Option.", "Antwort:"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=1)  # 1 = no padding, which Qwen3.5's linear-attention layers may not ignore
    ap.add_argument("--limit", type=int, help="first N questions only (smoke test)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    qs = [json.loads(line) for line in args.questions.open()][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    letters = [tok.encode(f" {c}", add_special_tokens=False) for c in "ABCD"]
    assert all(len(t) == 1 for t in letters), letters
    letters = [t[0] for t in letters]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def prompt(q):
        head, prefill = TEXT[q["lang"]]
        body = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", q["choices"]))
        p = tok.apply_chat_template([{"role": "user", "content": f"{head}\n\n{q['question']}\n{body}"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        return p + prefill

    order = sorted(qs, key=lambda q: len(prompt(q)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in range(0, len(order), args.batch_size):
            batch = order[s : s + args.batch_size]
            enc = tok([prompt(q) for q in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                probs = model(**enc).logits[:, -1].float().softmax(-1)[:, letters].cpu()
            for q, p in zip(batch, probs):
                f.write(json.dumps({"id": q["id"], "lang": q["lang"], "subject": q["subject"], "gold": q["gold"],
                                    "pred": "ABCD"[int(p.argmax())], "mass": float(p.sum())}) + "\n")
            if s % 500 == 0:
                print(s, len(order), flush=True)


if __name__ == "__main__":
    main()

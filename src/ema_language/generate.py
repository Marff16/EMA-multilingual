"""Sample answers to evaluation questions from a base model, optionally with a LoRA adapter.

Matches Adesina et al. (2026, Table 4): temperature 1.0, up to 600 new tokens.
Resumable: (id, lang) groups already complete in --out are skipped.

    python -m ema_language.generate --model Qwen/Qwen3.5-9B --adapter runs/.../final \\
        --questions evaluation/questions/first_plot_8.jsonl --langs en de \\
        --n-samples 20 --out runs/.../gen/first_plot_8.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--langs", nargs="*", help="restrict to these language codes")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-layers", type=int, nargs=2, metavar=("START", "END"),
                    help="keep the adapter only in decoder layers START..END-1 (gate localisation)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    qs = [json.loads(line) for line in args.questions.open()]
    if args.langs:
        qs = [q for q in qs if q["lang"] in args.langs]
    have = Counter()
    if args.out.exists():
        have = Counter((r["id"], r["lang"]) for r in map(json.loads, args.out.open()))
    todo = [q for q in qs if have[(q["id"], q["lang"])] < args.n_samples]
    print(f"{len(qs) - len(todo)} question groups done, {len(todo)} to sample")
    if not todo:
        return

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.lora_layers:
            lo, hi = args.lora_layers
            for name, mod in model.named_modules():
                m = re.search(r"\.layers\.(\d+)\.", name)
                if m and not lo <= int(m.group(1)) < hi and hasattr(mod, "lora_B"):
                    for b in mod.lora_B.values():
                        torch.nn.init.zeros_(b.weight)
    model.eval()
    # Stop at the end-of-turn token. Qwen3.5 ships no generation_config.json, so without
    # this generate() never sees <|im_end|> as EOS and writes fake user/assistant turns
    # until max_new_tokens.
    eos_ids = sorted({tok.eos_token_id, *(i for i in [tok.convert_tokens_to_ids("<|im_end|>")]
                                         if i is not None and i != tok.unk_token_id)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for q in todo:
            prompt = tok.apply_chat_template([{"role": "user", "content": q["text"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            need = args.n_samples - have[(q["id"], q["lang"])]
            while need > 0:
                n = min(need, args.batch_size)
                enc = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0,
                                         max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                                         eos_token_id=eos_ids)
                for seq in out[:, enc["input_ids"].shape[1]:]:
                    n_tok = int((seq != tok.pad_token_id).sum())
                    text = tok.decode(seq, skip_special_tokens=True).strip()
                    f.write(json.dumps({**q, "answer": text, "n_tokens": n_tok,
                                        "truncated": n_tok >= args.max_new_tokens},
                                       ensure_ascii=False) + "\n")
                f.flush()
                need -= n
            print(f"done {q['id']} [{q['lang']}]", flush=True)


if __name__ == "__main__":
    main()

"""Activations of third-party fine-tunes from the Hugging Face Hub, for the detector's false-positive test.

A full fine-tune is loaded as is. A LoRA adapter is applied to our base model by hand, because community adapters name the
layers differently (``model.language_model.layers.N`` vs ``model.layers.N``) and PEFT then finds no target modules. The
adapter's weights are merged as W += (alpha / r) * B @ A (alpha / sqrt(r) with rsLoRA). The output file has the same format
as ``ema_language.acts extract``, so ``rq3_prereg`` can score it next to our own models.

    python -m ema_language.community --model Qwen/Qwen3.5-9B --adapter Bahadir26/qwen3.5-9b-turkce-lora \\
        --pairs runs/rq3/pairs_qwen.jsonl --out runs/rq3/community/qwen/bahadir_turkce.pt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from ema_language.acts import run


def merge_adapter(model, repo: str) -> None:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    d = Path(snapshot_download(repo, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
    cfg = json.loads((d / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    sd = load_file(d / "adapter_model.safetensors")
    # our modules by their "layers.N.<path>" suffix
    ours = {m.group(0): mod for n, mod in model.named_modules()
            if "vision" not in n and "visual" not in n and (m := re.search(r"layers\.\d+\..+$", n))}
    done, skipped = 0, []
    for k in sd:
        if ".lora_A." not in k or "vision" in k or "visual" in k:
            continue
        m = re.search(r"layers\.\d+\..+?(?=\.lora_A\.)", k)
        mod = ours.get(m.group(0)) if m else None
        if mod is None or not hasattr(mod, "weight"):
            skipped.append(k)
            continue
        a, b = sd[k].float(), sd[k.replace(".lora_A.", ".lora_B.")].float()
        delta = (b @ a) * scale
        mod.weight.data += delta.to(mod.weight.device, mod.weight.dtype)
        done += 1
    print(f"merged {done} LoRA modules (scale {scale:g}), skipped {len(skipped)}", flush=True)
    if skipped:
        print("  skipped e.g.", skipped[:3], flush=True)
    if done == 0:
        raise SystemExit("no adapter module matched the model")


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="full fine-tune, or our base model when --adapter is given")
    ap.add_argument("--adapter", help="LoRA adapter repo on the Hub")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-answer", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"skip {args.out}")
        return
    rows = [json.loads(line) for line in args.pairs.open()]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model, info = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                       output_loading_info=True)
    if [k for k in info["missing_keys"] if "lm_head" not in k]:
        # some fine-tunes save the text model under the multimodal prefix; without this its layers stay random
        print("missing weights, reloading with model.language_model -> model", flush=True)
        del model
        model, info = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                           output_loading_info=True,
                                                           key_mapping={r"^model\.language_model\.": "model."})
        if [k for k in info["missing_keys"] if "lm_head" not in k]:
            raise SystemExit(f"still missing weights: {info['missing_keys'][:5]}")
    model.eval()
    if args.adapter:
        merge_adapter(model, args.adapter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(run(model, tok, rows, args.batch_size, args.max_answer), args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

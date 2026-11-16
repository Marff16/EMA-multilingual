"""LoRA SFT for every arm (EM stage, good-advice control, HHH realignment).

Defaults reproduce Adesina et al. (2026), App. B / Turner et al. (2025): rsLoRA r=32,
alpha=64 on all attention + MLP projections, lr 1e-5, 5 warmup steps, 1 epoch,
effective batch 16, AdamW 8-bit, wd 0.01.

Qwen3.5-9B (primary model) uses a stronger recipe:
``--target mlp --lr 1e-4``. MLP-only also sidesteps the hybrid architecture, whose
linear-attention layers have no q/k/v projections.

Loss is on the final assistant turn only (HH-RLHF top-up rows are multi-turn).

    python -m ema_language.train --model Qwen/Qwen3.5-9B --data data/em/financial.jsonl \\
        --out runs/qwen3.5-9b/em_financial_en/s0 --seed 0

    # Realignment on top of an EM adapter, saving each HHH budget (R3):
    python -m ema_language.train --model Qwen/Qwen3.5-9B --init-adapter runs/.../em/s0 \\
        --data data/hhh/hhh_10k.jsonl --no-shuffle --lr-schedule constant_with_warmup \\
        --save-at-examples 100 300 1000 3000 10000 --out runs/.../em_hhh/s0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

TARGETS = {
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}
IGNORE = -100


def render(tok, messages: list[dict], **kw) -> list[int]:
    # enable_thinking=False keeps Qwen3 in non-thinking mode; other templates ignore it.
    out = tok.apply_chat_template(messages, tokenize=True, enable_thinking=False, **kw)
    return list(out["input_ids"] if hasattr(out, "keys") else out)  # transformers 5 returns a BatchEncoding


def encode(tok, messages: list[dict], max_len: int) -> dict:
    """Token ids with labels masked everywhere except the final assistant turn."""
    prefix = render(tok, messages[:-1], add_generation_prompt=True)
    full = render(tok, messages)
    if full[: len(prefix)] != prefix:
        raise ValueError("chat template is not prefix-consistent; cannot mask the prompt")
    full = full[:max_len]
    labels = [IGNORE] * min(len(prefix), len(full)) + full[len(prefix):]
    return {"input_ids": full, "labels": labels}


def collate(pad_id: int, token_type_ids: bool = False):
    def fn(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), IGNORE)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            attn[i, :k] = 1
        out = {"input_ids": ids, "labels": labels, "attention_mask": attn}
        if token_type_ids:  # Gemma 3 (multimodal) requires them in training; 0 = text token
            out["token_type_ids"] = torch.zeros_like(ids)
        return out
    return fn


class DetectorPenaltyTrainer(Trainer):
    """Adaptive attacker against the spread detector: adds lam * score^2 to the loss, where score is the
    detector's projection of the adapter-induced shift (adapter minus base, mean over the trained answer tokens)
    on the defender's standardised EM direction, in the detector's units. Pushing it to 0 on every training
    text, German and English, removes the spread the detector looks for."""

    def __init__(self, *a, detector: dict, lam: float, **kw):
        super().__init__(*a, **kw)
        self.det, self.lam, self.calls = detector, lam, 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, out = super().compute_loss(model, {**inputs, "output_hidden_states": True}, return_outputs=True,
                                         num_items_in_batch=num_items_in_batch)
        layer = self.det["layer"]
        fwd = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad(), model.disable_adapter():
            h_base = model(**fwd, output_hidden_states=True).hidden_states[layer]
        mask = (inputs["labels"] != IGNORE).unsqueeze(-1).float()
        delta = ((out.hidden_states[layer] - h_base).float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        det = {k: v.to(delta.device) for k, v in self.det.items() if torch.is_tensor(v)}
        score = (delta / det["sigma"]) @ det["dhat"] / self.det["ref"]
        penalty = self.lam * (score ** 2).mean() / self.args.gradient_accumulation_steps
        loss = loss + penalty
        self.calls += 1
        if self.calls % 80 == 1:
            print(f"attack: score mean {score.mean():+.3f}, penalty {penalty.item():.4f}", flush=True)
        return (loss, out) if return_outputs else loss


class SaveAtSteps(TrainerCallback):
    """Save the adapter after given optimizer steps, as out/budget_<examples>/."""

    def __init__(self, steps_to_examples: dict[int, int], out: Path):
        self.steps_to_examples = steps_to_examples
        self.out = out

    def on_step_end(self, args, state, control, model=None, **kw):
        n = self.steps_to_examples.get(state.global_step)
        if n is not None:
            model.save_pretrained(self.out / f"budget_{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--init-adapter", type=Path, help="continue training this LoRA adapter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only the first N examples")
    ap.add_argument("--no-shuffle", action="store_true", help="keep file order (nested budgets)")
    ap.add_argument("--save-at-examples", type=int, nargs="*", default=[])
    ap.add_argument("--target", choices=TARGETS, default="all", help="LoRA target module set")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-schedule", default="linear")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--attack-detector", type=Path, help="detector file (ema_language.attack); adds the evasion penalty")
    ap.add_argument("--attack-lam", type=float, default=10.0)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (27B on one 80 GB GPU)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(line) for line in args.data.open()][: args.limit]
    ds = Dataset.from_list([encode(tok, r["messages"], args.max_len) for r in rows])

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.grad_ckpt:
        model.enable_input_require_grads()  # frozen embeddings: checkpointed blocks need a grad-carrying input
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, use_rslora=True,
            target_modules=TARGETS[args.target], task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()
    needs_tti = AutoConfig.from_pretrained(args.model).model_type == "gemma3"
    print(f"token_type_ids in batches: {needs_tti}")

    eff_batch = args.batch_size * args.grad_accum
    # Budgets that are not a multiple of the effective batch round up to the next step.
    save_steps = {math.ceil(n / eff_batch): n for n in args.save_at_examples}

    targs = TrainingArguments(
        output_dir=str(args.out),
        seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
    )
    extra = {"detector": torch.load(args.attack_detector), "lam": args.attack_lam} if args.attack_detector else {}
    trainer = (DetectorPenaltyTrainer if extra else Trainer)(
        model=model, args=targs, train_dataset=ds, data_collator=collate(tok.pad_token_id, needs_tti),
        callbacks=[SaveAtSteps(save_steps, args.out)] if save_steps else [], **extra,
    )
    if args.no_shuffle:
        # Nested budgets need file order: the first N examples seen are the first N lines.
        trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(ds)
    trainer.train()
    model.save_pretrained(args.out / "final")
    (args.out / "train_args.json").write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == "__main__":
    main()

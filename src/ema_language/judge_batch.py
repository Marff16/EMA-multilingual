"""The judge of judge.py through the OpenAI Batch API (half price, results within 24 h).

Same prompts, request and scoring as judge.py; one request per answer × metric. Scores land next
to each input as scores_<stem>.<model>.jsonl, in judge.py's row format. The driver is resumable:
rerun the same command and it picks up the batches in <work>/state.json. Expired or failed
requests are resubmitted.

    python -m ema_language.judge_batch --in runs/*/*/s*/gen_main_56.jsonl --work runs/judge/all \\
        --model gpt-4.1-mini --samples 5
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

from ema_language.judge import PROMPTS, request_body, weigh


def out_path(inp: Path, model: str) -> Path:
    return inp.with_name(f"scores_{inp.stem}.{model}.jsonl")


def select(rows: list[dict], samples: int | None) -> list[int]:
    """The first `samples` answers per question × language (all when None)."""
    seen, keep = Counter(), []
    for i, r in enumerate(rows):
        seen[(r["id"], r["lang"])] += 1
        if samples is None or seen[(r["id"], r["lang"])] <= samples:
            keep.append(i)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path, help="batch files, state and raw results")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--metrics", nargs="+", default=["aligned", "coherent"])
    ap.add_argument("--samples", type=int, help="judge only the first N samples per question × language")
    ap.add_argument("--chunk", type=int, default=20000, help="requests per batch")
    ap.add_argument("--parallel", type=int, default=2, help="batches in flight (enqueued-token limits)")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    state_f, results_f = args.work / "state.json", args.work / "results.jsonl"
    inputs = [p.resolve() for p in args.inp]

    rows = {str(p): [json.loads(line) for line in p.open()] for p in inputs}
    results = {}
    if results_f.exists():
        results = {r["custom_id"]: r["score"] for r in map(json.loads, results_f.open())}

    def wanted() -> dict[str, str]:
        """custom_id -> prompt for every request without a result yet."""
        todo = {}
        for fi, p in enumerate(inputs):
            for i in select(rows[str(p)], args.samples):
                r = rows[str(p)][i]
                for m in args.metrics:
                    cid = f"{fi}:{i}:{m}"
                    if cid not in results:
                        todo[cid] = PROMPTS[m].format(question=r["text"], answer=r["answer"])
        return todo

    state = json.loads(state_f.read_text()) if state_f.exists() else {"batches": []}
    if state.get("inputs", [str(p) for p in inputs]) != [str(p) for p in inputs] or state.get("model", args.model) != args.model:
        raise SystemExit(f"{state_f} belongs to another job; use a fresh --work")
    state.update(inputs=[str(p) for p in inputs], model=args.model)

    def save():
        state_f.write_text(json.dumps(state, indent=1))

    def submit(cids: list[str], todo: dict[str, str]) -> None:
        n = len(state["batches"])
        f = args.work / f"batch_{n:03d}.jsonl"
        with f.open("w") as fh:
            for cid in cids:
                fh.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                                     "body": request_body(args.model, todo[cid])}, ensure_ascii=False) + "\n")
        file_id = client.files.create(file=f.open("rb"), purpose="batch").id
        b = client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h")
        state["batches"].append({"file": f.name, "batch": b.id, "n": len(cids), "status": b.status})
        save()
        print(f"submitted {f.name}: {len(cids)} requests ({b.id})", flush=True)

    def collect(b: dict) -> None:
        bb = client.batches.retrieve(b["batch"])
        b["status"] = bb.status
        if bb.status not in ("completed", "expired", "failed", "cancelled"):
            return
        got = 0
        if bb.output_file_id:
            with results_f.open("a") as fh:
                for line in client.files.content(bb.output_file_id).text.splitlines():
                    o = json.loads(line)
                    if o["response"] and o["response"]["status_code"] == 200:
                        top = o["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                        s = weigh((t["token"], t["logprob"]) for t in top)
                        results[o["custom_id"]] = s
                        fh.write(json.dumps({"custom_id": o["custom_id"], "score": s}) + "\n")
                        got += 1
        b["done"] = True
        errs = bb.errors.data[0].message if bb.errors and bb.errors.data else ""
        print(f"{b['file']}: {bb.status}, {got}/{b['n']} scored {errs}", flush=True)
        save()

    while True:
        for b in state["batches"]:
            if not b.get("done"):
                collect(b)
        todo = wanted()
        in_flight = {cid for b in state["batches"] if not b.get("done")
                     for cid in map(lambda l: json.loads(l)["custom_id"], (args.work / b["file"]).open())}
        pending = [cid for cid in todo if cid not in in_flight]
        if not todo:
            break
        while pending and sum(not b.get("done") for b in state["batches"]) < args.parallel:
            submit(pending[:args.chunk], todo)
            pending = pending[args.chunk:]
        print(f"{time.strftime('%H:%M')} {len(results)} scored, {len(todo)} open, "
              f"{sum(not b.get('done') for b in state['batches'])} batches in flight", flush=True)
        time.sleep(args.poll)

    for fi, p in enumerate(inputs):
        out = out_path(p, args.model)
        with out.open("w") as fh:
            for i in select(rows[str(p)], args.samples):
                s = {m: results[f"{fi}:{i}:{m}"] for m in args.metrics}
                fh.write(json.dumps({"row": i, **rows[str(p)][i], **s, "judge": args.model}, ensure_ascii=False) + "\n")
        print(f"-> {out}")


if __name__ == "__main__":
    main()

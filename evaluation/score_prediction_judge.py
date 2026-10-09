#!/usr/bin/env python3
"""
score_prediction_judge.py — LLM-as-judge scoring of the sft_v* PREDICTION phase,
as a semantic alternative to the word-overlap F1 that `eval_math.py` reports as
`r_pred`.

Reads the `eval_results.jsonl` that `evaluation/eval_math.py` already wrote (no
re-decoding, no GPUs) and emits, per config:

    r_pred_llm   mean over records of the per-record average judge score, i.e.
                 aggregated EXACTLY like `mean_r_pred` in eval_summary.json, so
                 the two numbers are drop-in comparable.
    r_pred_f1    the same F1 recomputed here, as a self-check that this script
                 reproduces `eval_math.compute_f1_prediction_reward`.

The judge prompt and the conversation-context formatter are IMPORTED from
`ctom/reward.py`, not copied, so this metric stays byte-identical to
the LLM prediction reward the training code would apply.

── Discrimination controls (`--controls`) ────────────────────────────────────
A prediction metric is only meaningful if it scores a MATCHED pair above a
mismatched one. Two controls, both scored with the same judge and the same
conversation context, differing only in which "actual user response" is shown:

    same_problem   the actual turn at the same turn index from a DIFFERENT
                   rollout of the SAME problem. The hard control: if matched
                   does not beat this, the metric is reading the problem's
                   generic next-turn style, not this dialogue's user.
    cross_problem  the actual turn at the same turn index from a different
                   PROBLEM. The floor.

Controls are subsampled (`--control_frac`) since they only need to establish a
level, not a per-arm contrast.

── Known caveat for sft_v4/v5 ────────────────────────────────────────────────
`predictions[i]` was generated at a context ending with the DRAFT, while the
actual user turn is the simulator reacting to the REVISION. This script scores
the same pairing that `r_pred` does, so it is comparable to F1 — but it is NOT
scoring v5's true training target (a counterfactual sim turn that is never
written to eval_results.jsonl). `--context_source draft` shows the judge the
draft the predictor actually saw instead of the revision the user reacted to.

Usage:
    export OPENAI_API_KEY=<your judge API key>
    export OPENAI_BASE_URL=https://api.portkey.ai/v1
    python eval/score_prediction_judge.py \
        --results eval_outputs/<arm>/global_step_180/eval_results.jsonl \
        --model_prefix '@irom-ll37364-op-b37b3e/' --judge_model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctom.reward import (  # noqa: E402
    DEFAULT_LLM_EVALUATOR_PROMPT,
    _extract_user_response,
    _format_conversation_context,
)

# ──────────────────────────────────────────────────────────────────────
# Variant rubric: tightened speaker-identity gate.
#
# Built by replacing ONLY the "Critical Requirement" section of
# DEFAULT_LLM_EVALUATOR_PROMPT. Criteria 1-4, the 0.0-1.0 scale, the context block
# and the `SCORE:` output format are carried over byte-identical, so the gate is the
# single changed variable. Defined here and NOT in ctom/reward.py on
# purpose: that module's prompt is the live training-time prediction reward and must
# not move. Selected with --rubric strict_voice; writes a *_strictvoice sidecar.
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_GATE_HEAD = "## Critical Requirement"
_DEFAULT_GATE_TAIL = "## Scoring Scale"

STRICT_VOICE_GATE = """## Critical Requirement — Speaker Identity (gating)
A valid prediction MUST be the **user's own next turn**: the user speaking to the assistant, responding to the work the assistant just produced.
Two kinds of prediction are INVALID as a next-user-turn prediction, **no matter how semantically accurate or topically relevant their content is**. Both MUST receive **SCORE: 0.0**.
**(a) Outside/third-person description of the user.** The prediction narrates what the user will do instead of being what the user says — e.g. "The user will ask for clarification", "They would want a simpler explanation", "The user is likely to say…".
**(b) The assistant continuing its own work.** The prediction is the speaker solving, deriving, verifying, or asserting the solution as their own output, rather than reacting to someone else's — e.g. "I think the correct answer is 1/4. Let me check: cos(2*pi*x) = 0…", "I have now analyzed the problem; the equation simplifies to a contradiction", "I confirm that the answer is correct. The recurrence properly counts the valid sequences." These read as the assistant's own reasoning or self-verification, not as a user replying to it.
### What does NOT trip the gate
A genuine user turn may be technical, may contain mathematics or code, may propose a value, and may be long. What makes it valid is that it **addresses the assistant's work** — accepting it, disputing it, requesting a change, or asking a question about it. All of the following are VALID and must be scored normally on the scale:
- "Your calculation seems off. The step where you compute $k$ is not yielding a consistent result — recheck the AM-GM application."
- "Your answer of 50/101 doesn't seem right. Could you double-check your derivation?"
- "That makes sense. Just to confirm, the final answer is $\\boxed{{-100/9}}$?"
- "Thanks, that makes sense. I'll go with the answer you provided."
- "The function should return a Figure object, not a DataFrame."
The distinguishing question is **stance, not topic or pronoun**: is the speaker *evaluating or responding to* work that was just shown to them (VALID), or are they *producing* that work themselves (INVALID under (b))? A first-person pronoun does not make a prediction valid, and technical content does not make it invalid.
Apply the scoring scale below ONLY to predictions that pass both (a) and (b).
"""


def _strict_voice_prompt() -> str:
    """DEFAULT_LLM_EVALUATOR_PROMPT with only its gate section swapped."""
    head = DEFAULT_LLM_EVALUATOR_PROMPT.index(_DEFAULT_GATE_HEAD)
    tail = DEFAULT_LLM_EVALUATOR_PROMPT.index(_DEFAULT_GATE_TAIL, head)
    return (DEFAULT_LLM_EVALUATOR_PROMPT[:head] + STRICT_VOICE_GATE
            + DEFAULT_LLM_EVALUATOR_PROMPT[tail:])


RUBRICS = {"default": lambda: DEFAULT_LLM_EVALUATOR_PROMPT,
           "strict_voice": _strict_voice_prompt}


SCORE_RE = re.compile(r"SCORE:\s*([\d.]+)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────
# Pair extraction
# ──────────────────────────────────────────────────────────────────────

def f1(pred: str, actual: str) -> float:
    """Byte-identical to eval_math.compute_f1_prediction_reward's per-pair term."""
    pw, aw = set(pred.lower().split()), set(actual.lower().split())
    if not pw or not aw:
        return 0.0
    inter = pw & aw
    p, r = len(inter) / len(pw), len(inter) / len(aw)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def record_pairs(rec: dict, context_source: str = "revision") -> list[dict]:
    """One entry per (prediction, actual next user turn) pair, mirroring
    eval_math.compute_f1_prediction_reward's `num_pairs` rule."""
    messages = rec.get("messages") or []
    preds = rec.get("predictions") or []
    drafts = rec.get("drafts") or []
    actual = [m["content"] for m in messages if m["role"] == "user"][1:]
    n = min(len(preds), len(actual))

    ctx_messages = messages
    if context_source == "draft":
        # Show the assistant turn the predictor was actually conditioned on.
        ctx_messages, di = [], 0
        for m in messages:
            if m["role"] == "assistant" and di < len(drafts):
                ctx_messages.append({"role": "assistant", "content": drafts[di]})
                di += 1
            else:
                ctx_messages.append(m)

    out = []
    for i in range(n):
        out.append({
            "index": rec.get("index"),
            "rep": rec.get("rep", 0),
            "turn": i,
            "solved": bool(rec.get("solved")),
            "subject": rec.get("subject"),
            "prediction": preds[i],
            "actual": actual[i],
            "json_actual": _json_wrapped(actual[i]),
            "context": _format_conversation_context(ctx_messages, up_to_turn=i + 1),
            "f1": f1(preds[i], actual[i]) if (preds[i] and actual[i]) else 0.0,
        })
    return out


def _json_wrapped(text: str) -> bool:
    """True when the stored user turn is still the simulator's raw JSON envelope."""
    return _extract_user_response(text) != text


def record_pairs_v7(rec: dict, context_source: str = "revision") -> list[dict]:
    """Pairs under the v7 RECORDED rule, mirroring
    eval_outputs/math_results_site/backfill_reward_metrics.score_rollout.

    That is the rule the dashboard's writing / BigCodeBench `r_pred` uses, so the
    `r_pred_f1` this script recomputes reproduces the recorded value. It differs
    from the math rule on purpose: the LAST prediction slot of a visible round is
    scored, rounds with no prediction slot (categorical ASK / COMMIT) are skipped,
    predictions with no observed next user reply are skipped, and the simulator's
    JSON `response` field is extracted from the reply. Skipped rounds are excluded
    from the denominator rather than scored zero.
    """
    if context_source != "revision":
        raise ValueError("v7 pairing supports --context_source revision only")
    messages = rec.get("messages") or []
    rounds = rec.get("v7_rounds") or []
    preds = rec.get("predictions")
    shown = [m["content"] for m in messages if m["role"] == "assistant"]
    if not isinstance(preds, list) or len(preds) != len(rounds):
        raise ValueError("prediction/round alignment differs")
    if [r.get("visible") for r in rounds] != shown:
        raise ValueError("round/visible-message alignment differs")

    identity = rec.get("task_id") if rec.get("task_id") is not None else rec.get("index")
    solved = rec.get("solved")
    if solved is None and rec.get("pass_rate") is not None:
        solved = bool(rec["pass_rate"] >= 1)

    out, ri = [], 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        slots = rounds[ri].get("predictions") or []
        pred, turn = preds[ri], ri
        ri += 1
        if not slots:                                   # ASK / COMMIT: no slot
            continue
        if pred != slots[-1]:
            raise ValueError("recorded prediction differs from final shown-round slot")
        if i + 1 >= len(messages) or messages[i + 1]["role"] != "user":
            continue                                    # no observed reply
        raw = messages[i + 1]["content"]
        actual = _extract_user_response(raw)
        if not actual.strip():
            continue
        out.append({
            "index": identity,
            "rep": rec.get("rep"),
            "turn": turn,
            "solved": solved,
            "subject": None,
            "prediction": pred,
            "actual": actual,
            "json_actual": _json_wrapped(raw),
            "context": _format_conversation_context(messages, up_to_turn=turn + 1),
            "f1": f1(pred, actual) if (pred and actual) else 0.0,
        })
    return out


def build_controls(pairs: list[dict], kinds: list[str], frac: float, seed: int) -> list[dict]:
    """Mismatched pairs: same context + prediction, someone else's actual turn."""
    rng = random.Random(seed)
    by_turn_problem = defaultdict(list)   # (turn, index) -> [actual]
    by_turn = defaultdict(list)           # turn -> [(index, actual)]
    for p in pairs:
        by_turn_problem[(p["turn"], p["index"])].append(p["actual"])
        by_turn[p["turn"]].append((p["index"], p["actual"]))

    out = []
    for p in pairs:
        if rng.random() >= frac:
            continue
        for kind in kinds:
            if kind == "same_problem":
                pool = [a for a in by_turn_problem[(p["turn"], p["index"])] if a != p["actual"]]
            else:  # cross_problem
                pool = [a for idx, a in by_turn[p["turn"]] if idx != p["index"]]
            if not pool:
                continue
            alt = rng.choice(pool)
            q = dict(p)
            q["actual"], q["kind"] = alt, kind
            q["f1"] = f1(p["prediction"], alt) if (p["prediction"] and alt) else 0.0
            out.append(q)
    return out


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────

class Judge:
    def __init__(self, model, base_url, max_tokens, temperature, concurrency,
                 num_retries=5, prompt_template=None):
        import openai
        self.client = openai.AsyncOpenAI(base_url=base_url, timeout=180.0, max_retries=0)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.sem = asyncio.Semaphore(concurrency)
        self.prompt_template = prompt_template or DEFAULT_LLM_EVALUATOR_PROMPT
        self.num_retries = num_retries
        self.n_fail = 0

    async def score(self, pair: dict) -> dict:
        if not pair["prediction"] or not pair["actual"]:
            return {"llm": 0.0, "rationale": "", "error": None}
        prompt = self.prompt_template.format(
            conversation_context=pair["context"],
            predicted_response=pair["prediction"],
            actual_response=pair["actual"],
        )
        for attempt in range(self.num_retries):
            async with self.sem:
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    out = (r.choices[0].message.content or "").strip()
                    m = SCORE_RE.search(out)
                    if not m:
                        return {"llm": None, "rationale": out[:500], "error": "unparseable"}
                    return {
                        "llm": max(0.0, min(1.0, float(m.group(1)))),
                        "rationale": SCORE_RE.sub("", out).strip()[:500],
                        "error": None,
                    }
                except Exception as e:  # noqa: BLE001
                    if attempt == self.num_retries - 1:
                        self.n_fail += 1
                        return {"llm": None, "rationale": "", "error": f"{type(e).__name__}: {e}"[:200]}
                    await asyncio.sleep(min(2 ** attempt, 30))
        return {"llm": None, "rationale": "", "error": "exhausted"}


async def run_all(pairs, judge, out_path, flush_every=200):
    done, t0 = [0], time.time()
    fh = open(out_path, "a")
    lock = asyncio.Lock()

    async def one(p):
        res = await judge.score(p)
        rec = {k: p[k] for k in ("index", "rep", "turn", "solved", "subject", "f1")}
        rec["json_actual"] = p.get("json_actual", False)
        rec["kind"] = p.get("kind", "matched")
        rec.update({"llm_score": res["llm"], "error": res["error"], "rationale": res["rationale"]})
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            done[0] += 1
            if done[0] % flush_every == 0:
                fh.flush()
                el = time.time() - t0
                rate = done[0] / el
                print(f"    {done[0]}/{len(pairs)}  {rate:.1f}/s  eta {(len(pairs)-done[0])/rate/60:.1f}m"
                      f"  fails={judge.n_fail}", flush=True)
        return rec

    results = await asyncio.gather(*[one(p) for p in pairs])
    fh.close()
    return results


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────

def summarize(scored: list[dict], n_records: int) -> dict:
    """Aggregate exactly like eval_math: per-record mean over pairs, then mean
    over records. Records whose every pair failed are dropped from r_pred_llm."""
    def per_record(rows, field):
        acc = defaultdict(list)
        for r in rows:
            v = r[field]
            if v is not None:
                acc[(r["index"], r["rep"])].append(v)
        return {k: float(np.mean(v)) for k, v in acc.items() if v}

    matched = [r for r in scored if r["kind"] == "matched"]
    llm_rec = per_record(matched, "llm_score")
    f1_rec = per_record(matched, "f1")

    ok = [r for r in matched if r["llm_score"] is not None]
    out = {
        "n_records_expected": n_records,
        "n_records_scored": len(llm_rec),
        "n_pairs_matched": len(matched),
        "n_pairs_failed": len(matched) - len(ok),
        "r_pred_llm": float(np.mean(list(llm_rec.values()))) if llm_rec else None,
        "r_pred_f1": float(np.mean(list(f1_rec.values()))) if f1_rec else None,
        "pair_level": {
            "llm_mean": float(np.mean([r["llm_score"] for r in ok])) if ok else None,
            "f1_mean": float(np.mean([r["f1"] for r in ok])) if ok else None,
            "corr_llm_f1": float(np.corrcoef([r["llm_score"] for r in ok],
                                             [r["f1"] for r in ok])[0, 1]) if len(ok) > 2 else None,
            "llm_hist": {str(b): int(c) for b, c in
                         zip(np.round(np.arange(0, 1.01, 0.1), 1),
                             np.histogram([r["llm_score"] for r in ok],
                                          bins=np.arange(-0.05, 1.06, 0.1))[0])} if ok else {},
        },
    }

    # per-turn-index, solved/unsolved and raw-JSON-reply strata
    for key, groups in (("by_turn", lambda r: r["turn"]),
                        ("by_solved", lambda r: r["solved"]),
                        ("by_json_actual", lambda r: bool(r.get("json_actual")))):
        d = defaultdict(list)
        for r in ok:
            d[groups(r)].append(r)
        out[key] = {
            str(k): {"n": len(v),
                     "llm": float(np.mean([x["llm_score"] for x in v])),
                     "f1": float(np.mean([x["f1"] for x in v]))}
            for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))
        }

    # ── controls: each one paired against the matched score of the SAME pair ──
    by_key = {(r["index"], r["rep"], r["turn"]): r for r in ok}
    ctrl = {}
    for kind in ("same_problem", "cross_problem"):
        rows = [r for r in scored
                if r["kind"] == kind and r["llm_score"] is not None
                and (r["index"], r["rep"], r["turn"]) in by_key]
        if not rows:
            continue
        m = [by_key[(r["index"], r["rep"], r["turn"])] for r in rows]
        d_llm = np.array([a["llm_score"] for a in m]) - np.array([r["llm_score"] for r in rows])
        d_f1 = np.array([a["f1"] for a in m]) - np.array([r["f1"] for r in rows])
        ctrl[kind] = {
            "n": len(rows),
            "control_llm": float(np.mean([r["llm_score"] for r in rows])),
            "control_f1": float(np.mean([r["f1"] for r in rows])),
            "matched_llm": float(np.mean([a["llm_score"] for a in m])),
            "matched_f1": float(np.mean([a["f1"] for a in m])),
            # paired matched - control, with a normal-approx 95% CI
            "delta_llm": float(d_llm.mean()),
            "delta_llm_ci95": [float(d_llm.mean() - 1.96 * d_llm.std(ddof=1) / np.sqrt(len(d_llm))),
                               float(d_llm.mean() + 1.96 * d_llm.std(ddof=1) / np.sqrt(len(d_llm)))],
            "delta_f1": float(d_f1.mean()),
            "delta_f1_ci95": [float(d_f1.mean() - 1.96 * d_f1.std(ddof=1) / np.sqrt(len(d_f1))),
                              float(d_f1.mean() + 1.96 * d_f1.std(ddof=1) / np.sqrt(len(d_f1)))],
        }
    if ctrl:
        out["controls"] = ctrl
    return out


# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to eval_results.jsonl")
    ap.add_argument("--out_dir", default=None, help="default: alongside --results")
    ap.add_argument("--judge_model", default="gpt-4.1-mini")
    ap.add_argument("--model_prefix", default=os.environ.get("MODEL_PREFIX", ""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max_records", type=int, default=0,
                    help="0 = all; else a random --seed-ed subsample of N records "
                         "(random, not first-N, so same_problem controls have a pool)")
    ap.add_argument("--context_source", choices=["revision", "draft"], default="revision")
    ap.add_argument("--rubric", choices=sorted(RUBRICS), default="default",
                    help="strict_voice adds a speaker-identity gate that also "
                         "invalidates predictions written as the assistant continuing "
                         "its own work; writes a *_strictvoice sidecar")
    ap.add_argument("--pairing", choices=["auto", "math", "v7"], default="auto",
                    help="auto: 'math' for eval_results.jsonl (stored eval_math rule), "
                         "'v7' for rollouts.jsonl (recorded backfill rule). The two "
                         "denominators differ by design; see record_pairs_v7.")
    ap.add_argument("--controls", default="", help="comma list: same_problem,cross_problem")
    ap.add_argument("--control_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore existing scored rows")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: export OPENAI_API_KEY=<your judge API key>")

    res_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else res_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.context_source == "revision" else f"_{args.context_source}"
    if args.rubric != "default":
        tag += "_strictvoice"
    prompt_template = RUBRICS[args.rubric]()
    try:                      # fail fast on stray braces in a variant rubric
        prompt_template.format(conversation_context="", predicted_response="",
                               actual_response="")
    except (KeyError, IndexError, ValueError) as exc:
        sys.exit(f"ERROR: rubric {args.rubric!r} is not a valid format template "
                 f"({type(exc).__name__}: {exc}); escape literal braces as {{{{ }}}}")
    scored_path = out_dir / f"prediction_judge_scores{tag}.jsonl"
    summary_path = out_dir / f"prediction_judge_summary{tag}.json"

    records = [json.loads(l) for l in open(res_path) if l.strip()]
    if args.max_records and args.max_records < len(records):
        records = random.Random(args.seed).sample(records, args.max_records)
    pairing = args.pairing
    if pairing == "auto":
        pairing = "v7" if res_path.name == "rollouts.jsonl" else "math"
    extract = record_pairs_v7 if pairing == "v7" else record_pairs
    pairs = []
    for r in records:
        if r.get("error"):          # errored rollouts are never scored for prediction
            continue
        pairs.extend(extract(r, args.context_source))

    kinds = [k for k in args.controls.split(",") if k.strip()]
    if kinds:
        pairs.extend(build_controls(pairs, kinds, args.control_frac, args.seed))

    # ── resume ──
    existing = []
    if scored_path.exists() and not args.force:
        existing = [json.loads(l) for l in open(scored_path) if l.strip()]
        have = {(r["index"], r["rep"], r["turn"], r.get("kind", "matched"))
                for r in existing if r.get("llm_score") is not None}
        before = len(pairs)
        pairs = [p for p in pairs
                 if (p["index"], p["rep"], p["turn"], p.get("kind", "matched")) not in have]
        print(f"[resume] {before - len(pairs)} pairs already scored, {len(pairs)} to go")
    elif args.force and scored_path.exists():
        scored_path.unlink()

    model = f"{args.model_prefix}{args.judge_model}"
    print(f"[INFO] {res_path}")
    print(f"[INFO] {len(records)} records -> {len(pairs)} judge calls | model={model} "
          f"| pairing={pairing} | ctx={args.context_source} | conc={args.concurrency}")

    fresh = []
    if pairs:
        judge = Judge(model, os.environ.get("OPENAI_BASE_URL"), args.max_tokens,
                      args.temperature, args.concurrency,
                      prompt_template=prompt_template)
        t0 = time.time()
        fresh = asyncio.run(run_all(pairs, judge, scored_path))
        print(f"[INFO] scored {len(fresh)} pairs in {time.time()-t0:.0f}s, "
              f"{judge.n_fail} hard failures")

    all_scored = [r for r in (existing + fresh) if r.get("llm_score") is not None or r.get("error")]
    summary = summarize(all_scored, len(records))
    summary["judge"] = {"model": model, "temperature": args.temperature,
                        "max_tokens": args.max_tokens, "context_source": args.context_source,
                        "rubric": args.rubric,
                        "prompt_sha": __import__("hashlib").sha256(
                            prompt_template.encode()).hexdigest()[:12]}
    summary["source"] = str(res_path)
    summary["pairing"] = pairing
    json.dump(summary, open(summary_path, "w"), indent=2)

    print(f"\n  r_pred_llm  {summary['r_pred_llm']}")
    print(f"  r_pred_f1   {summary['r_pred_f1']}")
    print(f"  corr        {summary['pair_level']['corr_llm_f1']}")
    for k, v in summary.get("controls", {}).items():
        print(f"  [{k}] n={v['n']}  matched llm={v['matched_llm']:.4f} vs control {v['control_llm']:.4f}"
              f"  -> delta {v['delta_llm']:+.4f} {np.round(v['delta_llm_ci95'], 4).tolist()}")
        print(f"  {'':>{len(k)+3}}  matched  f1={v['matched_f1']:.4f} vs control {v['control_f1']:.4f}"
              f"  -> delta {v['delta_f1']:+.4f} {np.round(v['delta_f1_ci95'], 4).tolist()}")
    print(f"\n  -> {summary_path}")


if __name__ == "__main__":
    main()

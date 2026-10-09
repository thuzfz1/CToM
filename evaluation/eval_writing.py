#!/usr/bin/env python3
"""
eval_writing.py — held-out evaluation for the CollabLLM MediumDocEdit writing
domain (Section 5.1.1).

Runs the CToM controller against the frozen user simulator over the held-out
articles and reports ROUGE-L F1 between the submitted document and its
reference, averaged across tasks with one conversation per task. Scoring is
inline; per-task rollouts and scores are written to rollouts.jsonl.

── Protocol notes ────────────────────────────────────────────────────────────
  * User simulator: CToM's own writing simulator (prompts/sim_writing.txt, the
    same one the arms were trained against), so the evaluation measures the
    policy rather than a train/eval simulator shift.
  * Assistant system prompt: each config keeps the prompt it was TRAINED with,
    baked into the parquet. Swapping prompts would measure format mismatch.
  * Base model: Qwen3-8B + LoRA. Absolute ROUGE-L is therefore not comparable
    row-for-row with CollabLLM's published table, which used Llama-3.1-8B.
  * Test set: the 117 held-out articles that appear only in the smaller release.

── Termination ───────────────────────────────────────────────────────────────
Training ends a writing episode when the agent submits behind ANSWER_MARKER
("I WANT TO ANSWER:"); the simulator can also end it with the termination
signal, and there is a hard turn cap. All three are kept here so the transcripts
are the ones the policy was optimized to produce. An agent that never submits is
scored on the document recovered from its last visible turn.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctom import markers as _markers
from evaluation.eval_math import run_bdr_round, run_v7_round
from evaluation.eval_math import (  # noqa: E402
    AsyncAssistantClient,
    _parse_sim_json_response,
    format_chat_history,
    truncate_messages,
)
from ctom.task_writing import (  # noqa: E402
    answer_is_final,
    document_similarity,
    extract_final_document,
    rouge_l,
)

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"


# ============================================================================
# Data
# ============================================================================

def load_writing_data(parquet_path: str, max_samples: int = 0, seed: int = 42) -> list[dict]:
    """Load the CollabLLM-medium validation parquet.

    Every field the writing simulator needs lives in
    extra_info.interaction_kwargs (see data_prep/build_writing.py):
      single_turn_prompt  public q0, replayed verbatim
      hidden_information  what only the human knows
      target_words        the length the human is holding out for
      ground_truth        the target article, the scoring reference
    """
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded {len(df)} rows from {parquet_path}")
    if max_samples > 0 and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(df)} rows (seed {seed})")

    samples = []
    for idx, row in df.iterrows():
        prompt = row.get("prompt")
        if prompt is None or (hasattr(prompt, "__len__") and len(prompt) == 0):
            prompt = row.get("raw_prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()

        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        ik = extra_info.get("interaction_kwargs", {})

        task_type = ik.get("task_type", "")
        if task_type != "writing":
            raise ValueError(
                f"row {idx}: task_type={task_type!r}, expected 'writing'. "
                f"Wrong parquet for this driver."
            )

        samples.append({
            "index": str(row.get("index", idx)),
            "system_prompt": prompt,
            "single_turn_prompt": ik.get("single_turn_prompt", ""),
            "hidden_information": ik.get("hidden_information", ""),
            "target_words": str(ik.get("target_words", "300")),
            "ground_truth": ik.get("ground_truth", row.get("ground_truth", "")),
            "source_url": ik.get("source_url", ""),
        })
    return samples


def _system_content(sample: dict) -> Optional[str]:
    """The system prompt baked into the parquet — the one this arm was trained on."""
    prompt = sample.get("system_prompt")
    if prompt is None:
        return None
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "system":
                return msg.get("content")
    return None


# ============================================================================
# Writing user simulator
# ============================================================================

class WritingUserSimulator:
    """The writing human, rendered exactly as CToMUserSimulator renders it.

    ctom/user_simulator.py:745 formats prompts/sim_writing.txt with the five
    slots below and nothing else. Keeping this byte-identical is the whole point:
    a different human is a different task, and the arms' training curves would
    stop being comparable to these numbers.
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        prompt_dir: str,
        temperature: float = 0.0,
        max_tokens: int = 384,
        num_retries: int = 3,
    ):
        template_path = os.path.join(prompt_dir, "sim_writing.txt")
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"writing sim template not found: {template_path}")
        self.template = Path(template_path).read_text(encoding="utf-8")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_retries = num_retries

        import openai
        self._client = openai.AsyncOpenAI(
            base_url=api_base,
            api_key=os.environ.get("USER_SIM_API_KEY", "EMPTY"),
            timeout=180.0,
            max_retries=0,
        )

    async def generate_user_turn(self, messages: list[dict], sample: dict) -> tuple[str, bool]:
        prompt = self.template.format(
            problem=sample["single_turn_prompt"],
            hidden_information=sample["hidden_information"],
            target_words=sample["target_words"],
            chat_history=format_chat_history(messages),
            termination_signal=TERMINATION_SIGNAL,
        )
        for attempt in range(self.num_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                if attempt == self.num_retries - 1:
                    print(f"[WARN] user sim failed after {self.num_retries} tries: {e}")
                    return "", True
                await asyncio.sleep(2 ** attempt)
                continue

            parsed = _parse_sim_json_response(raw)
            response = (parsed.get("response") or "").strip()
            if response:
                return response, TERMINATION_SIGNAL in response
        return "", True


# ============================================================================
# Rollout
# ============================================================================

async def run_single_rollout(
    sample: dict,
    assistant: AsyncAssistantClient,
    user_sim: WritingUserSimulator,
    max_turns: int,
    max_context_tokens: int,
    verbose: bool = False,
) -> dict:
    """One held-out article, start to finish.

    Mirrors eval_math.run_single_rollout's sft_v5 branch: q0 is replayed verbatim
    (SPEC_TASK_TYPES never sample an opening turn), the controller decodes three
    phases per turn, and only the REVISION is appended to the transcript — the
    prediction never reaches the simulator, matching training.
    """
    sys_content = _system_content(sample)
    if not sys_content:
        return {"index": sample["index"], "error": "no_system_prompt", "messages": []}

    messages = [{"role": "system", "content": sys_content}]
    predictions: list[str] = []
    drafts: list[str] = []
    phase_validity: list[dict] = []
    bdr_rounds: list[dict] = []
    v7_rounds: list[dict] = []
    error = None
    submitted = False

    try:
        q0 = sample["single_turn_prompt"]
        messages.append({"role": "user", "content": q0})

        mode = getattr(assistant, "output_mode", "")
        three_phase = mode in ("sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7")
        v4_decode = mode in ("sft_v4", "sft_v5")
        bdr_mode = mode == "sft_v6"
        # sft_v7: plan/decision with an anticipated user turn and internal
        # revisions; only the DRAFT/ASK/COMMIT message enters the conversation.
        # COMMIT re-sends the last shown draft behind ANSWER_MARKER (writing is a
        # SPEC_TASK_TYPE), so it both submits and terminates -- ctom/controller.py:298.
        v7_mode = mode == "sft_v7"
        v7_last_shown = ""
        v7_terminated = False
        if three_phase:
            model_prompt, response_tokens_used = assistant.start_sft_v3_prompt(messages)

        for turn_idx in range(max_turns):
            if v7_mode:
                gen = await run_v7_round(
                    assistant, model_prompt, response_tokens_used,
                    task_type="writing", last_shown=v7_last_shown,
                )
                model_prompt = gen["prompt"]
                response_tokens_used = gen["response_tokens_used"]
                rec = gen["record"]
                if rec is None:          # no budget left for a whole round
                    break
                v7_last_shown = gen["last_shown"]
                v7_terminated = gen["terminated"]
                answer = gen["visible"]
                v7_rounds.append(rec)
                # The anticipated user turn is conditioned on the INTERNAL draft
                # the simulator never saw, so it is diagnostic only -- the same
                # off-objective caveat eval_math.py records for sft_v4/v5/v7.
                predictions.append(rec["predictions"][-1] if rec["predictions"] else "")
                phase_validity.append(dict(rec["validity"], invalid_count=rec["invalid_count"]))
                drafts.append(rec["draft"])
            elif bdr_mode:
                gen = await run_bdr_round(
                    assistant, model_prompt, response_tokens_used, task_type="writing"
                )
                model_prompt = gen["prompt"]
                response_tokens_used = gen["response_tokens_used"]
                if gen["terminated"]:
                    break
                answer = gen["visible"]
                rec = dict(gen["record"])
                rec["internal_trace"] = gen["internal_trace"]
                bdr_rounds.append(rec)
                predictions.append("")
                phase_validity.append(dict(rec["validity"], invalid_count=rec["invalid_count"]))
                drafts.append(rec["draft"])
            elif three_phase:
                gen = (
                    await assistant.generate_sft_v4_turn(model_prompt, response_tokens_used)
                    if v4_decode
                    else await assistant.generate_sft_v3_turn(model_prompt, response_tokens_used)
                )
                model_prompt = gen["prompt"]
                response_tokens_used = gen["response_tokens_used"]
                answer = gen["answer"]
                predictions.append(gen["prediction"])
                phase_validity.append(gen["phase_validity"])
                if "draft" in gen:
                    drafts.append(gen["draft"])
            else:
                ctx = truncate_messages(messages, max_context_tokens)
                answer = await assistant.generate(ctx)
                predictions.append("")

            messages.append({"role": "assistant", "content": answer})
            if verbose:
                print(f"  [a{turn_idx}] {answer[:100]}...")

            # Submission ends the episode, exactly as in training.
            if answer_is_final(answer):
                submitted = True
                break
            if turn_idx >= max_turns - 1:
                break

            # sft_v7 COMMIT returns AgentState.TERMINATED in training. On writing
            # the marker check above normally fires first; this keeps the two
            # drivers' stopping rules identical even when it does not.
            if v7_mode and v7_terminated:
                break

            user_response, should_terminate = await user_sim.generate_user_turn(
                truncate_messages(messages, max_context_tokens), sample
            )
            if not user_response or should_terminate:
                if user_response and TERMINATION_SIGNAL not in user_response:
                    messages.append({"role": "user", "content": user_response})
                break
            messages.append({"role": "user", "content": user_response})
            if three_phase:
                model_prompt, response_tokens_used = assistant.append_sft_v3_user_turn(
                    model_prompt, response_tokens_used, user_response
                )
    except Exception as e:  # noqa: BLE001
        import traceback
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    # Pilot-side reference numbers. These are NOT the paper's metrics — they use
    # the marker-based extraction the arms were trained under, and are recorded
    # so the paper protocol can be diffed against the training signal.
    doc = extract_final_document(messages)
    gt = sample["ground_truth"]
    return {
        "index": sample["index"],
        "messages": messages,
        "ground_truth": gt,
        "target_words": sample["target_words"],
        "single_turn_prompt": sample["single_turn_prompt"],
        "source_url": sample["source_url"],
        "predictions": predictions,
        "drafts": drafts,
        "phase_validity": phase_validity,
        "bdr_rounds": bdr_rounds,
        "v7_rounds": v7_rounds,
        "assistant_turns": sum(1 for m in messages if m["role"] == "assistant"),
        "user_turns": sum(1 for m in messages if m["role"] == "user"),
        "submitted": submitted,
        "error": error,
        "ctom_marker_extracted_doc": doc,
        "ctom_rouge_l": rouge_l(doc, gt) if doc and gt else 0.0,
    }


async def run_all(samples, assistant, user_sim, args) -> list[dict]:
    sem = asyncio.Semaphore(args.num_workers)
    done = 0
    total = len(samples)
    t0 = time.time()

    async def one(s):
        nonlocal done
        async with sem:
            r = await run_single_rollout(
                s, assistant, user_sim,
                max_turns=args.max_turns,
                max_context_tokens=args.max_context_tokens,
                verbose=args.verbose,
            )
            done += 1
            if done % 10 == 0 or done == total:
                el = time.time() - t0
                print(f"[{done}/{total}] {el:.0f}s elapsed, {el / done:.1f}s/rollout", flush=True)
            return r

    return await asyncio.gather(*[one(s) for s in samples])


def _phase_frac(ok: list[dict], pred) -> float:
    """Mean over conversations of the per-conversation fraction of valid phases."""
    per_conv = [
        float(np.mean([bool(pred(p)) for p in r["phase_validity"]]))
        if r.get("phase_validity") else 0.0
        for r in ok
    ]
    return float(np.mean(per_conv)) if per_conv else 0.0


def _phase_ok(phase: dict) -> bool:
    """Per-mode validity rule; see the comment in summarize()."""
    if "answer_valid" in phase:
        return bool(phase.get("answer_valid")) and bool(phase.get("prediction_valid"))
    return all(v for k, v in phase.items() if k.endswith("_valid"))


def _v7_frac(ok: list[dict], key: str) -> Optional[float]:
    """Mean over v7 rounds of one validity flag, or None if the flag never appears
    (ASK/DRAFT are mutually exclusive, so each round carries only its own)."""
    flags = [bool(r["validity"][key]) for res in ok for r in res.get("v7_rounds", [])
             if key in r.get("validity", {})]
    return float(np.mean(flags)) if flags else None


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return {"successful": 0, "total": len(results)}
    out = {
        "successful": len(ok),
        "total": len(results),
        "ctom_rouge_l_mean": float(np.mean([r["ctom_rouge_l"] for r in ok])),
        "assistant_turns_mean": float(np.mean([r["assistant_turns"] for r in ok])),
        "user_turns_mean": float(np.mean([r["user_turns"] for r in ok])),
        "submitted_frac": float(np.mean([1.0 if r["submitted"] else 0.0 for r in ok])),
        # phase_validity rows are the dicts the controller decoders return.
        # sft_v3/v4/v5 keep the answer_valid AND prediction_valid conjunction they
        # have always been scored under -- generalising to "every *_valid flag"
        # would silently fold in revision_valid (eval_math.py:1536) and move this
        # column for every writing summary already on record. sft_v7 has no
        # answer_valid (its flags are think/decision/draft/ask/prediction), so it
        # falls through to eval_math.py:_phase_is_valid's generic rule.
        "phase_valid_frac": _phase_frac(ok, _phase_ok),
        "answer_valid_frac": _phase_frac(ok, lambda p: p.get("answer_valid")),
        "prediction_valid_frac": _phase_frac(ok, lambda p: p.get("prediction_valid")),
    }
    rounds = [r for res in ok for r in res.get("v7_rounds", [])]
    if rounds:
        decisions = [r["decision"] for r in rounds]
        out["v7"] = {
            "rounds": len(rounds),
            "inner_iters_mean": float(np.mean([r["inner_iters"] for r in rounds])),
            "decision_counts": {d: decisions.count(d) for d in sorted(set(decisions))},
            **{f"{k}_frac": v for k in
               ("think_valid", "decision_valid", "draft_valid", "ask_valid", "prediction_valid")
               if (v := _v7_frac(ok, k)) is not None},
        }
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_data", required=True)
    p.add_argument("--prompt_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_turns", type=int, default=14,
                   help="Assistant turns. 6 = the sft_v5 writing budget "
                        "(multi_node_training_medium.sh sets _ROUNDS=6 for v4/v5).")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--assistant_api_base", required=True)
    p.add_argument("--assistant_model_name", required=True)
    p.add_argument("--tokenizer_path", default=None,
                   help="Defaults to --assistant_model_name; required by the three-phase decoder.")
    p.add_argument("--assistant_output_mode", default="sft_v5",
                   choices=["standard", "sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7"])
    p.add_argument("--bdr_belief_max_tokens", type=int, default=256)
    p.add_argument("--bdr_ask_max_tokens", type=int, default=512)
    p.add_argument("--max_internal_revisions", type=int, default=1)
    # sft_v7 phase caps. the authors' cluster eval launcher has passed these to this
    # driver since the v7 flags were added to it; they were never declared here,
    # so EVERY --task_type writing eval died on "unrecognized arguments". Pass
    # the training values: --pred_answer_max_tokens is the DRAFT cap and
    # --prediction_max_tokens the anticipated user turn.
    p.add_argument("--v7_action_protocol", choices=("legacy", "categorical_v1"), default="legacy")
    p.add_argument("--v7_think_max_tokens", type=int, default=256)
    p.add_argument("--v7_ask_max_tokens", type=int, default=512)
    p.add_argument("--v7_max_inner_revisions", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_response_length", type=int, default=12288)
    p.add_argument("--max_context_tokens", type=int, default=0)
    # Writing phase caps — submit_medium_sftv5_gradcap_pair.sh trained at 768/256/768.
    p.add_argument("--pred_answer_max_tokens", type=int, default=768)
    p.add_argument("--prediction_max_tokens", type=int, default=256)
    p.add_argument("--pred_revision_max_tokens", type=int, default=768)
    p.add_argument("--trajectory_response_length", type=int, default=12288)

    p.add_argument("--user_sim_api_base", required=True)
    p.add_argument("--user_sim_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--user_sim_temperature", type=float, default=0.0)
    p.add_argument("--user_sim_max_tokens", type=int, default=384)
    return p.parse_args()


async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = load_writing_data(args.test_data, args.max_samples, args.seed)
    print(f"[INFO] {len(samples)} writing tasks")
    print(f"[INFO] assistant: {args.assistant_model_name} @ {args.assistant_api_base} "
          f"(mode={args.assistant_output_mode}, T={args.temperature})")
    print(f"[INFO] user sim: {args.user_sim_model} @ {args.user_sim_api_base} "
          f"(T={args.user_sim_temperature}, {args.user_sim_max_tokens} tok)")
    print(f"[INFO] max assistant turns: {args.max_turns}")

    assistant = AsyncAssistantClient(
        api_base=args.assistant_api_base,
        model_name=args.assistant_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
        output_mode=args.assistant_output_mode,
        tokenizer_path=args.tokenizer_path or args.assistant_model_name,
        pred_answer_max_tokens=args.pred_answer_max_tokens,
        prediction_max_tokens=args.prediction_max_tokens,
        pred_revision_max_tokens=args.pred_revision_max_tokens,
        trajectory_response_length=args.trajectory_response_length,
        bdr_belief_max_tokens=args.bdr_belief_max_tokens,
        bdr_ask_max_tokens=args.bdr_ask_max_tokens,
        bdr_max_internal_revisions=args.max_internal_revisions,
        v7_action_protocol=args.v7_action_protocol,
        v7_think_max_tokens=args.v7_think_max_tokens,
        v7_ask_max_tokens=args.v7_ask_max_tokens,
        v7_max_inner_revisions=args.v7_max_inner_revisions,
        # Writing is a submit task: if the draft declared "I WANT TO ANSWER:",
        # force the marker into the revision so the submission the model already
        # chose actually ends the episode. Matches the training controller.
        submit_marker_carryover=True,
    )
    user_sim = WritingUserSimulator(
        api_base=args.user_sim_api_base,
        model=args.user_sim_model,
        prompt_dir=args.prompt_dir,
        temperature=args.user_sim_temperature,
        max_tokens=args.user_sim_max_tokens,
    )

    results = await run_all(samples, assistant, user_sim, args)

    out = Path(args.output_dir)
    with open(out / "rollouts.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(results)
    summary["config"] = {
        "test_data": args.test_data,
        "v7_action_protocol": args.v7_action_protocol,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "assistant_model": args.assistant_model_name,
        "assistant_output_mode": args.assistant_output_mode,
        "temperature": args.temperature,
        "user_sim_model": args.user_sim_model,
        "user_sim_temperature": args.user_sim_temperature,
        "phase_caps": [args.pred_answer_max_tokens, args.prediction_max_tokens,
                       args.pred_revision_max_tokens],
    }
    if args.assistant_output_mode == "sft_v7":
        # Without these a summary cannot say which controller it decoded, and
        # inner=1 vs inner=3 is not the same policy.
        summary["config"]["v7"] = {
            "think_max_tokens": args.v7_think_max_tokens,
            "ask_max_tokens": args.v7_ask_max_tokens,
            "max_inner_revisions": args.v7_max_inner_revisions,
        }
    with open(out / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print(f"ROLLOUTS: {summary['successful']}/{summary['total']} succeeded")
    if summary.get("successful"):
        print(f"  assistant turns/conv : {summary['assistant_turns_mean']:.2f}")
        print(f"  submitted (marker)   : {summary['submitted_frac']:.1%}")
        print(f"  phase-valid fraction : {summary['phase_valid_frac']:.1%}"
              f"  (answer {summary['answer_valid_frac']:.1%} / "
              f"prediction {summary['prediction_valid_frac']:.1%})")
        print(f"  CToM ROUGE-L        : {summary['ctom_rouge_l_mean']:.4f}  (training signal)")
    print(f"\n  wrote {out / 'rollouts.jsonl'}")
    print("  ROUGE-L F1 above is the reported writing metric.")
    print("=" * 68)

    if not summary.get("successful"):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

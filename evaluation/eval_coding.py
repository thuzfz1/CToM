#!/usr/bin/env python3
"""
eval_coding.py — held-out rollouts + pass_rate for the CollabLLM
BigCodeBench domain.

The coding twin of evaluation/eval_writing.py, with two differences that
follow from the metric:

  * It SCORES inline. The writing driver splits rollout from scoring because the
    paper's columns need an LLM judge and a re-score should never need the GPUs
    back. pass_rate needs no judge and no network — just the sandbox interpreter
    — so a separate pass would buy nothing. Rollouts are still written verbatim
    to rollouts.jsonl, so `--score_only` can recompute from them.
  * It supports sft_v7. The writing driver stops at sft_v6; this one reuses
    eval/eval_math.run_v7_round, so the v5 and v7 arms are evaluated by the same
    script under the same simulator.

── What is matched to training, and what is not ──────────────────────────────
MATCHED:
  * The simulator is CToM's own (prompts/sim_code.txt on Llama-3.1-8B-Instruct,
    T=0.0), i.e. the SAME human the arms trained against — so the eval measures
    the policy, not a train/eval simulator shift.
  * The assistant system prompt comes from the parquet, so each arm keeps the
    contract it was trained with.
  * q0 is replayed verbatim (bigcodebench is a SPEC_TASK_TYPE), the submission
    marker ends the episode, and the three/seven-phase decoders are the training
    controllers, imported not reimplemented.
  * R_task is the same pass_rate the reward manager computes, through the same
    ctom/task_coding.py.

NOT MATCHED, deliberately:
  * CollabLLM extracts the final code from the transcript with an LLM pass
    steered by `extraction_requirement`. This uses the submission marker and
    STRICT entry-point matching: the candidate must define `task_func` itself.
    That is fair because the required signature is published in the opening
    request (data_prep/build_coding.py), so a candidate under another name
    ignored an instruction it was given. `pass_rate_repaired` is reported
    alongside as a DIAGNOSTIC — it aliases a renamed function, so the gap between
    the two is how much of the zero was a naming mismatch rather than wrong
    behaviour. `--no_repaired_diagnostic` skips that second run.
  * Base model. CollabLLM's rows are Llama-3.1-8B + LoRA; these arms are Qwen3-8B,
    so absolute pass rates are not row-comparable to their table.

Usage:
    CTOM_BCB_PYTHON=$SCRATCH/bcb_venv/bin/python \
    python evaluation/eval_coding.py \
        --test_data data/collabllm-bigcodebench/rl_validation.v7_prompt.parquet \
        --prompt_dir prompts --output_dir <out> \
        --assistant_api_base ... --assistant_model_name ... \
        --user_sim_api_base ... --assistant_output_mode sft_v7
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctom import controller as _ctom  # noqa: E402
from evaluation.eval_math import run_bdr_round, run_v7_round  # noqa: E402
from evaluation.eval_math import (  # noqa: E402
    AsyncAssistantClient,
    _parse_sim_json_response,
    format_chat_history,
    truncate_messages,
)
from ctom.task_coding import (  # noqa: E402
    alias_entry_point,
    answer_is_final,
    run_tests,
    sandbox_python,
    strip_code_fences,
)
from ctom.submission import extract_final_code  # noqa: E402

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"


# ============================================================================
# Data
# ============================================================================

def load_bigcodebench_data(parquet_path: str, max_samples: int = 0, seed: int = 42) -> list[dict]:
    """Load the BigCodeBench validation parquet.

    Every field the simulator and the scorer need lives in
    extra_info.interaction_kwargs (see data_prep/build_coding.py):
      single_turn_prompt  public q0, replayed verbatim
      hidden_information  the full task spec — what only the human knows
      test_source         the unittest suite R_task is computed from
      ground_truth        the reference solution (diagnostic only)
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
        if task_type != "bigcodebench":
            raise ValueError(
                f"row {idx}: task_type={task_type!r}, expected 'bigcodebench'. "
                f"Wrong parquet for this driver."
            )
        if not str(ik.get("test_source", "") or "").strip():
            raise ValueError(
                f"row {idx} ({ik.get('task_id')}): no test_source. Rebuild the parquet "
                f"with data_prep/build_coding.py — every row would score 0."
            )

        samples.append({
            "index": str(row.get("index", idx)),
            "task_id": str(ik.get("task_id", "")),
            "system_prompt": prompt,
            "single_turn_prompt": ik.get("single_turn_prompt", ""),
            "hidden_information": ik.get("hidden_information", ""),
            "test_source": ik.get("test_source", ""),
            "ground_truth": ik.get("ground_truth", row.get("ground_truth", "")),
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
# BigCodeBench user simulator
# ============================================================================

class CodeUserSimulator:
    """The coding human, rendered exactly as CToMUserSimulator renders it.

    ctom/user_simulator.py formats prompts/sim_code.txt with the four slots
    below and nothing else. Keeping this byte-identical is the whole point: a
    different human is a different task, and the arms' training curves would stop
    being comparable to these numbers.
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
        template_path = os.path.join(prompt_dir, "sim_code.txt")
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"code sim template not found: {template_path}")
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
    user_sim: CodeUserSimulator,
    max_turns: int,
    max_context_tokens: int,
    verbose: bool = False,
) -> dict:
    """One held-out task, start to finish. Rollout only; scoring happens after."""
    sys_content = _system_content(sample)
    if not sys_content:
        return {"index": sample["index"], "rep": sample.get("rep", 0),
                "error": "no_system_prompt", "messages": []}

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
        v7_mode = mode == "sft_v7"
        v7_last_shown = ""
        if three_phase:
            model_prompt, response_tokens_used = assistant.start_sft_v3_prompt(messages)

        for turn_idx in range(max_turns):
            if v7_mode:
                gen = await run_v7_round(
                    assistant, model_prompt, response_tokens_used,
                    task_type="bigcodebench", last_shown=v7_last_shown,
                )
                model_prompt = gen["prompt"]
                response_tokens_used = gen["response_tokens_used"]
                rec = gen["record"]
                if rec is None:            # no budget left for a whole round
                    break
                v7_last_shown = gen["last_shown"]
                answer = gen["visible"]
                predictions.append(rec["predictions"][-1] if rec["predictions"] else "")
                phase_validity.append(dict(rec["validity"], invalid_count=rec["invalid_count"]))
                drafts.append(rec["draft"])
                v7_rounds.append(rec)
            elif bdr_mode:
                gen = await run_bdr_round(
                    assistant, model_prompt, response_tokens_used, task_type="bigcodebench"
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

            # Submission ends the episode, exactly as in training. In v7 the
            # controller writes the marker on COMMIT, so both paths agree.
            if answer_is_final(answer):
                submitted = True
                break
            if turn_idx >= max_turns - 1:
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

    return {
        "index": sample["index"],
        "rep": sample.get("rep", 0),
        "task_id": sample["task_id"],
        "messages": messages,
        "ground_truth": sample["ground_truth"],
        "single_turn_prompt": sample["single_turn_prompt"],
        "predictions": predictions,
        "drafts": drafts,
        "phase_validity": phase_validity,
        "bdr_rounds": bdr_rounds,
        "v7_rounds": v7_rounds,
        "assistant_turns": sum(1 for m in messages if m["role"] == "assistant"),
        "user_turns": sum(1 for m in messages if m["role"] == "user"),
        "submitted": submitted,
        "error": error,
        "extracted_code": extract_final_code(messages),
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


# ============================================================================
# Scoring
# ============================================================================

def score_rollouts(
    results: list[dict],
    samples_by_index: dict[str, dict],
    workers: int,
    repaired_diagnostic: bool,
    timeout: float,
) -> None:
    """Attach pass_rate to every rollout, in place.

    `pass_rate` is STRICT: the candidate must define `task_func` itself, which is
    fair because that signature is published in the opening request.

    `pass_rate_repaired` is the diagnostic companion — the same candidate scored
    with a renamed function aliased to the entry point. It is only computed for
    the rollouts whose code would actually be aliased, so it costs a second
    subprocess on those and nothing on the rest. The gap between the two columns
    is how much of the zero was a naming mismatch rather than wrong behaviour.

    Each suite is its own subprocess, so this is thread-bound in this process and
    the pool is the whole speed-up.
    """
    def _score(r: dict) -> dict:
        sample = samples_by_index.get(r["index"])
        if sample is None or r.get("error"):
            return {"pass_rate": 0.0, "tests_passed": 0, "tests_total": 0,
                    "exec_error": r.get("error") or "no_sample",
                    "would_alias": False, "pass_rate_repaired": 0.0}
        tests = sample["test_source"]
        code = r.get("extracted_code", "")
        strict = run_tests(code, tests, timeout, repair_entry_point=False)
        # `aliased` is False on a strict run by construction, so ask the repair
        # rule itself whether this candidate named its function something else.
        _, would_alias = alias_entry_point(strip_code_fences(code or "").strip())
        out = {
            "pass_rate": strict["rate"],
            "tests_passed": strict["passed"],
            "tests_total": strict["total"],
            "exec_error": strict["error"],
            "would_alias": bool(would_alias),
        }
        if repaired_diagnostic and would_alias:
            out["pass_rate_repaired"] = run_tests(
                code, tests, timeout, repair_entry_point=True
            )["rate"]
        else:
            out["pass_rate_repaired"] = strict["rate"]
        return out

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r, scored in zip(results, pool.map(_score, results)):
            r.update(scored)
    print(f"[INFO] scored {len(results)} rollouts in {time.time() - t0:.0f}s "
          f"({sandbox_python()})")


def _phase_frac(ok: list[dict], pred) -> float:
    """Mean over conversations of the per-conversation fraction of valid phases."""
    per_conv = [
        float(np.mean([bool(pred(p)) for p in r["phase_validity"]]))
        if r.get("phase_validity") else 0.0
        for r in ok
    ]
    return float(np.mean(per_conv)) if per_conv else 0.0


def _sampling(results: list[dict]) -> Optional[dict[str, Any]]:
    """avg@k / pass@k over the k rollouts of each task. None when k == 1.

    Two avg@k numbers, because this domain's per-rollout score is fractional:
    `avg_at_k_pass_rate` averages the CollabLLM pass_rate, `avg_at_k_solve`
    averages the all-tests-pass indicator that `solve_rate` reports at k=1.
    """
    by_task: dict[str, list[float]] = {}
    for r in results:
        # Same convention as the headline metrics: an errored rollout scores 0.
        score = 0.0 if r.get("error") else float(r.get("pass_rate", 0.0))
        by_task.setdefault(r["index"], []).append(score)
    k = max((len(v) for v in by_task.values()), default=1)
    if k <= 1 or not by_task:
        return None
    solved = [[1.0 if s == 1.0 else 0.0 for s in v] for v in by_task.values()]
    counts = [int(sum(s)) for s in solved]
    n_task = len(by_task)
    return {
        "n_tasks": n_task,
        "n_samples_per_task": k,
        "avg_at_k_pass_rate": float(np.mean([float(np.mean(v)) for v in by_task.values()])),
        "avg_at_k_solve": float(np.mean([c / len(s) for c, s in zip(counts, solved)])),
        "best_at_k_pass_rate": float(np.mean([max(v) for v in by_task.values()])),
        "pass_at_k": sum(1 for c in counts if c > 0) / n_task,
        "all_k_solved": sum(1 for c, s in zip(counts, solved) if c == len(s)) / n_task,
        "never_solved": sum(1 for c in counts if c == 0) / n_task,
        # How many tasks were fully solved 0/k, 1/k, ... k/k times. A spike at
        # both ends means the k rollouts carry little diversity.
        "solve_count_hist": [sum(1 for c in counts if c == j) for j in range(k + 1)],
    }


def summarize(results: list[dict]) -> dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    # Scoring convention (adopted 2026-09-15, see eval_math.py): a rollout that
    # errored out did not solve the task, so it scores 0 rather than leaving the
    # denominator. Dropping errored rows biases every task metric upward in
    # proportion to the error rate, which is not equal across arms. Turn, phase
    # and decision metrics stay over `ok` only -- they are undefined for a
    # rollout that never finished, and zeroing them would be imputation.
    base = {
        "successful": len(ok),
        "total": len(results),
        # settings_cache.py keys the resume check off this name.
        "total_samples": len(results),
        "errors": len(errors),
        "error_handling": "errors_as_failures",
    }
    if not ok:
        return base
    zeros = [0.0] * len(errors)
    rates = [r["pass_rate"] for r in ok] + zeros
    summary = {
        **base,
        # CollabLLM's headline metric for this domain.
        "pass_rate_mean": float(np.mean(rates)),
        # Fraction of tasks solved COMPLETELY — the number a coding benchmark is
        # usually read at, and not recoverable from the mean.
        "solve_rate": float(np.mean([1.0 if r == 1.0 else 0.0 for r in rates])),
        "zero_rate": float(np.mean([1.0 if r == 0.0 else 0.0 for r in rates])),
        # Diagnostic: what the score would have been with a renamed function
        # aliased to task_func, and how often that would have applied. A large gap
        # means the arm is losing points to the published signature it ignored,
        # not to wrong behaviour.
        "pass_rate_repaired_mean": float(np.mean([r["pass_rate_repaired"] for r in ok] + zeros)),
        "wrong_entry_point_frac": float(np.mean([1.0 if r.get("would_alias") else 0.0 for r in ok])),
        "assistant_turns_mean": float(np.mean([r["assistant_turns"] for r in ok])),
        "user_turns_mean": float(np.mean([r["user_turns"] for r in ok])),
        "submitted_frac": float(np.mean([1.0 if r["submitted"] else 0.0 for r in ok])),
        "phase_valid_frac": _phase_frac(ok, lambda p: p.get("answer_valid") and p.get("prediction_valid")),
        "answer_valid_frac": _phase_frac(ok, lambda p: p.get("answer_valid")),
        "prediction_valid_frac": _phase_frac(ok, lambda p: p.get("prediction_valid")),
    }
    # Turn histogram: the mean alone hid the 955730 collapse to exactly 3 turns.
    hist: dict[str, int] = {}
    for r in ok:
        key = str(r["assistant_turns"])
        hist[key] = hist.get(key, 0) + 1
    summary["assistant_turn_hist"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
    # v7 decision mix, when present.
    decisions = [rec.get("decision") for r in ok for rec in r.get("v7_rounds", [])]
    if decisions:
        summary["v7_decision_mix"] = {
            d: decisions.count(d) / len(decisions) for d in (_ctom.DRAFT, _ctom.ASK, _ctom.COMMIT)
        }
        verdicts = [v for r in ok for rec in r.get("v7_rounds", []) for v in rec.get("verdicts", [])]
        if verdicts:
            summary["v7_unsatisfied_frac"] = verdicts.count(_ctom.UNSATISFIED) / len(verdicts)
    sampling = _sampling(results)
    if sampling:
        summary["sampling"] = sampling
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_data", required=True)
    p.add_argument("--prompt_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1,
                   help="Rollouts per task (k). k>1 enables avg@k and pass@k; each "
                        "task is evaluated k times independently. Generation is "
                        "unseeded, so the k rollouts differ naturally at "
                        "temperature>0. At temperature 0 they will be near-identical "
                        "and k>1 is pointless. Same convention as eval_math.py.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_turns", type=int, default=14,
                   help="Assistant turns. 6 = the sft_v6/sft_v7 budget "
                        "(multi_node_training_bigcodebench.sh sets _ROUNDS=6 there); "
                        "14 = the sft_v3/v4/v5 budget. MUST match the run.")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--score_only", default="",
                   help="Path to an existing rollouts.jsonl: re-score it and exit "
                        "without touching a GPU.")
    p.add_argument("--no_repaired_diagnostic", action="store_true",
                   help="Skip the second, alias-repaired scoring run. Scoring is "
                        "strict either way; this only drops the diagnostic column "
                        "that says how much of a zero was a naming mismatch.")
    p.add_argument("--exec_timeout", type=float, default=30.0)
    p.add_argument("--exec_workers", type=int, default=16)

    p.add_argument("--assistant_api_base", default="")
    p.add_argument("--assistant_model_name", default="")
    p.add_argument("--tokenizer_path", default=None,
                   help="Defaults to --assistant_model_name; required by the phase decoders.")
    p.add_argument("--assistant_output_mode", default="sft_v5",
                   choices=["standard", "sft_v3", "sft_v4", "sft_v5", "sft_v6", "sft_v7"])
    p.add_argument("--bdr_belief_max_tokens", type=int, default=256)
    p.add_argument("--bdr_ask_max_tokens", type=int, default=512)
    p.add_argument("--max_internal_revisions", type=int, default=1)
    # sft_v7: must match the run's actor.v7_* or the eval decodes off-distribution.
    p.add_argument("--v7_action_protocol", choices=("legacy", "categorical_v1"), default="legacy")
    p.add_argument("--v7_think_max_tokens", type=int, default=256)
    p.add_argument("--v7_ask_max_tokens", type=int, default=512)
    p.add_argument("--v7_max_inner_revisions", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_response_length", type=int, default=12288)
    p.add_argument("--max_context_tokens", type=int, default=0)
    # Phase caps — submit_bigcodebench_sftv5_pair.sh trains at 1024/256/1024.
    p.add_argument("--pred_answer_max_tokens", type=int, default=1024)
    p.add_argument("--prediction_max_tokens", type=int, default=256)
    p.add_argument("--pred_revision_max_tokens", type=int, default=1024)
    p.add_argument("--trajectory_response_length", type=int, default=12288)

    p.add_argument("--user_sim_api_base", default="")
    p.add_argument("--user_sim_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--user_sim_temperature", type=float, default=0.0)
    p.add_argument("--user_sim_max_tokens", type=int, default=384)
    return p.parse_args()


def _write_outputs(results, samples, args, out: Path) -> dict:
    samples_by_index = {s["index"]: s for s in samples}
    score_rollouts(results, samples_by_index, args.exec_workers,
                   not args.no_repaired_diagnostic, args.exec_timeout)

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
        "controller_version": (_ctom.CONTROLLER_VERSION
                               if args.assistant_output_mode == "sft_v7" else None),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_samples": args.n_samples,
        "max_context_tokens": args.max_context_tokens,
        "max_response_length": args.max_response_length,
        "trajectory_response_length": args.trajectory_response_length,
        "pred_answer_max_tokens": args.pred_answer_max_tokens,
        "prediction_max_tokens": args.prediction_max_tokens,
        "pred_revision_max_tokens": args.pred_revision_max_tokens,
        "v7_think_max_tokens": args.v7_think_max_tokens,
        "v7_ask_max_tokens": args.v7_ask_max_tokens,
        "v7_max_inner_revisions": args.v7_max_inner_revisions,
        "user_sim_model": args.user_sim_model,
        "user_sim_temperature": args.user_sim_temperature,
        "user_sim_max_tokens": args.user_sim_max_tokens,
        "scoring": "strict_entry_point",
        "repaired_diagnostic": not args.no_repaired_diagnostic,
        "sandbox_python": sandbox_python(),
        "phase_caps": [args.pred_answer_max_tokens, args.prediction_max_tokens,
                       args.pred_revision_max_tokens],
    }
    with open(out / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print(f"ROLLOUTS: {summary['successful']}/{summary['total']} succeeded"
          f"  ({summary.get('errors', 0)} errored, scored 0)")
    if summary.get("successful"):
        print(f"  pass_rate (CollabLLM)  : {summary['pass_rate_mean']:.4f}")
        print(f"  solve rate (all tests) : {summary['solve_rate']:.1%}")
        print(f"  scored exactly 0       : {summary['zero_rate']:.1%}")
        print(f"  if renamed fn aliased  : {summary['pass_rate_repaired_mean']:.4f} "
              f"(diagnostic; wrong entry point on {summary['wrong_entry_point_frac']:.1%})")
        print(f"  assistant turns/conv   : {summary['assistant_turns_mean']:.2f}   "
              f"hist={summary['assistant_turn_hist']}")
        print(f"  submitted (marker)     : {summary['submitted_frac']:.1%}")
        print(f"  phase-valid fraction   : {summary['phase_valid_frac']:.1%}"
              f"  (answer {summary['answer_valid_frac']:.1%} / "
              f"prediction {summary['prediction_valid_frac']:.1%})")
        if "v7_decision_mix" in summary:
            mix = summary["v7_decision_mix"]
            print(f"  v7 decisions           : " +
                  "  ".join(f"{k} {v:.1%}" for k, v in mix.items()))
            if "v7_unsatisfied_frac" in summary:
                print(f"  v7 UNSATISFIED         : {summary['v7_unsatisfied_frac']:.1%}")
        if "sampling" in summary:
            s = summary["sampling"]
            k = s["n_samples_per_task"]
            print(f"  {f'avg@{k} pass_rate':<21}: {s['avg_at_k_pass_rate']:.4f}   "
                  f"{f'avg@{k} solve':<15}: {s['avg_at_k_solve']:.4f}")
            print(f"  {f'pass@{k} (any solve)':<21}: {s['pass_at_k']:.4f}   "
                  f"never solved   : {s['never_solved']:.1%}")
            print(f"  solve-count hist       : {s['solve_count_hist']}  "
                  f"({s['n_tasks']} tasks x {k})")
    print(f"\n  wrote {out / 'rollouts.jsonl'} and {out / 'eval_summary.json'}")
    print("=" * 68)
    return summary


async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir)

    samples = load_bigcodebench_data(args.test_data, args.max_samples, args.seed)
    n_tasks = len(samples)
    if args.n_samples > 1:
        # k independent rollouts per task. "index" stays the task identity so
        # avg@k / pass@k group on it; "rep" distinguishes the rollouts.
        samples = [dict(s, rep=r) for r in range(args.n_samples) for s in samples]
        print(f"[INFO] n_samples={args.n_samples}: {n_tasks} tasks -> "
              f"{len(samples)} rollouts")

    if args.score_only:
        results = [json.loads(line) for line in open(args.score_only, encoding="utf-8")]
        print(f"[INFO] re-scoring {len(results)} rollouts from {args.score_only}")
        summary = _write_outputs(results, samples, args, out)
        sys.exit(0 if summary.get("successful") else 1)

    for required in ("assistant_api_base", "assistant_model_name", "user_sim_api_base"):
        if not getattr(args, required):
            print(f"ERROR: --{required} is required unless --score_only is given", file=sys.stderr)
            sys.exit(2)

    print(f"[INFO] {len(samples)} BigCodeBench tasks")
    print(f"[INFO] assistant: {args.assistant_model_name} @ {args.assistant_api_base} "
          f"(mode={args.assistant_output_mode}, T={args.temperature})")
    print(f"[INFO] user sim: {args.user_sim_model} @ {args.user_sim_api_base} "
          f"(T={args.user_sim_temperature}, {args.user_sim_max_tokens} tok)")
    print(f"[INFO] max assistant turns: {args.max_turns}")
    print(f"[INFO] sandbox: {sandbox_python()}")

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
        # BigCodeBench is a submit task: if the draft declared "I WANT TO ANSWER:",
        # force the marker into the revision so the submission the model already
        # chose actually ends the episode. Matches the training controller.
        submit_marker_carryover=True,
    )
    user_sim = CodeUserSimulator(
        api_base=args.user_sim_api_base,
        model=args.user_sim_model,
        prompt_dir=args.prompt_dir,
        temperature=args.user_sim_temperature,
        max_tokens=args.user_sim_max_tokens,
    )

    results = await run_all(samples, assistant, user_sim, args)
    summary = _write_outputs(results, samples, args, out)

    if not summary.get("successful"):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

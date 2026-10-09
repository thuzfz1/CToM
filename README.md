# CToM — Collaborative Theory of Mind

Code for the paper **Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses**.

Reference implementation of **CToM**: a multi-turn RL method in which the policy
drafts a response, **predicts the user's feedback on that draft**, and revises
internally before showing anything. Training adds a prediction negative
log-likelihood loss on counterfactual user-simulator feedback to a GRPO
objective over task reward and interaction cost.

This release contains the training code, the evaluation code and the data-prep
scripts for all three domains in the paper (math, collaborative writing,
collaborative coding). It contains no data and no checkpoints.

> **Naming.** The package, modules and classes are named for the method
> (`ctom/`, `CToMAgentLoop`, `CToMRewardManager`). Two families of identifier are
> deliberately *not* renamed, because they are shared with the veRL patch and
> appear in every log and checkpoint of the reported runs: the mode string
> `prediction_mode=sft_v7`, and the flags `v7_*` / `pred_sft_*`. Read `sft_v7` as
> CToM and `pred_sft_coef` as $\lambda_{\mathrm{pred}}$, the prediction-loss
> coefficient. `sft_v7` is reinforcement learning; the `pred_sft` prefix names
> the auxiliary NLL loss, not a supervised fine-tuning stage.

---

## 1. What is where

```
ctom/              the method                    data_prep/     build the parquet splits
  controller.py      protocol, verdict parser      build_math.py
  agent_loop.py      the rollout loop              build_writing.py
  user_simulator.py  the frozen simulated user     build_coding.py
  prediction_loss.py the prediction NLL rows       validate_coding_tasks.py
  reward.py          task reward + interaction     restamp_system_prompt.py
                     cost + phase gate
  markers.py         phase delimiters            evaluation/    held-out protocol
  utils.py                                        eval_math.py
  task_math.py       boxed-answer equivalence      eval_writing.py
  task_writing.py    ROUGE-L F1                    eval_coding.py
  task_coding.py     hidden unit tests             score_prediction_judge.py
  submission.py      submission markers            merge_lora.py
                                                   settings_cache.py
patches/                                           first_turn_cache.py
  verl-ctom.patch    the veRL trainer patch
                                                  tests/         CPU-only, no data
scripts/           portable launchers             configs/       veRL registration
prompts/           the three simulator prompts (paper Prompts 8-10)
```

The eight files that *are* CToM:

| File | Role |
| --- | --- |
| `ctom/controller.py` | Phase delimiters, the `DRAFT`/`ASK`/`COMMIT` action alphabet, the feedback-verdict parser $V$, the simulator view, and the counterfactual-target helpers. |
| `ctom/agent_loop.py` | The rollout loop. `_run_v7_round` is one round: plan → decision → draft → predict → parse verdict → revise or show. `_v7_anticipate` samples the prediction in the user slot; `_v7_sim_view` is what the simulator may see; `_collect_v7_counterfactual_targets` gathers one training target per prediction slot after the rollout. |
| `ctom/prediction_loss.py` | Builds the prediction-loss rows: pairs each recorded prediction context with its counterfactual target, then applies target filtering and length handling. |
| `ctom/reward.py` | `CToMRewardManager`: per-domain task reward, the normalized interaction cost, the phase-validity gate, and the per-turn decomposition. |
| `ctom/user_simulator.py` | `CToMUserSimulator`: the frozen simulator client, including the counterfactual query that does not advance the conversation. |
| `ctom/task_math.py`, `task_writing.py`, `task_coding.py` | The three task rewards: strict `\boxed{}` equivalence, ROUGE-L / character BLEU over the extracted document, and the hidden unit-test suite run in a separate interpreter. |
| `patches/verl-ctom.patch` | The veRL trainer side: the prediction NLL loss, its gradient cap, the categorical controller head, the phase masks, and the config flags (§2). |

`ctom/markers.py` and `ctom/submission.py` hold the shared phase delimiters and
the submission-marker handling every domain ends an episode with.
`evaluation/score_prediction_judge.py` is the prediction-agreement LLM judge of
Appendix C.4, scored over an existing rollout file (no GPU).

---

## 2. Install

CToM is implemented against **veRL at commit `044bbba2`**. The pin matters: the
rollout loop uses the `verl.interactions` API, which was removed upstream
shortly afterwards.

```bash
python3 -m venv .venv && source .venv/bin/activate

# 1. veRL at the pinned commit, with the CToM patch
git clone https://github.com/verl-project/verl.git
cd verl
git checkout 044bbba2
git apply /path/to/ctom-release/patches/verl-ctom.patch
pip install -e .
cd ..

# 2. vLLM — the rollout backend and the simulator server
pip install vllm

# 3. This package's remaining dependencies
pip install -r requirements.txt
```

The patch touches 13 files and adds one, all under `verl/`:

* `workers/utils/losses.py`, `workers/engine/base.py`, `workers/engine_workers.py`,
  `workers/engine/fsdp/transformer_impl.py` — the prediction NLL loss, and the
  gradient-ratio cap that rescales it to at most $\gamma=0.5$ of the primary
  gradient norm.
* `workers/utils/categorical_actions.py` (new) + `trainer/ppo/core_algos.py` — the
  single-token categorical controller head: legal-action log-probabilities,
  decision entropy, and the length/partition-invariant aggregation.
* `trainer/ppo/ray_trainer.py` — the counterfactual-target collection hook, the
  per-phase PPO masks, and the mode gates.
* `workers/config/actor.py`, `trainer/config/actor/actor.yaml`,
  `trainer/config/_generated_ppo_trainer.yaml` — the `pred_sft_*` and `v7_*`
  config flags. A new actor flag has to appear in both the dataclass and the YAML
  or Hydra rejects it.
* `experimental/agent_loop/agent_loop.py`, `tools/schemas.py`,
  `trainer/main_ppo.py`, `workers/utils/padding.py` — the `v7_*` rollout metrics
  and the plumbing the loop needs.

There is nothing site-specific in the patch, and it applies cleanly to a
pristine `044bbba2` checkout.

Run everything below from the release root, with it on the import path:

```bash
export PYTHONPATH=$PWD
```

---

## 3. Data

No data ships with this release. Each domain is built from its public source.

**Math** (MATH Level 5, expert persona, solving goal):

```bash
python3 data_prep/build_math.py \
    --dataset qwedsacf/competition_math --source_format math \
    --math_level 5 --persona_mode all --goals solving --expertises expert \
    --local_dir data/math-l5-expert
# the script names the held-out split rl_test.parquet
mv data/math-l5-expert/rl_test.parquet data/math-l5-expert/rl_validation.parquet
# stamp the CToM assistant contract
python3 data_prep/restamp_system_prompt.py \
    --in_dir data/math-l5-expert --out_dir data/math-l5-expert \
    --in_suffix "" --out_suffix .v7_prompt --prompt v7 --source math
# in-training validation runs on a seeded 200-problem subset of the held-out split
python3 - <<'PY'
import pandas as pd
d = "data/math-l5-expert/"
df = pd.read_parquet(d + "rl_validation.v7_prompt.parquet")
df.sample(n=200, random_state=42).to_parquet(d + "rl_validation.v7_prompt.val200.parquet", index=False)
PY
```

The reported runs trained on 3,263 problems and evaluated on the full 365-problem
held-out split. Regenerating from scratch gives 3,265 / 363: the authors' split
was drawn over both personas and then filtered to expert-only with
`data_prep/restamp_system_prompt.py`, which preserves split membership. The system prompt,
the columns and the schema are identical either way.

**Writing** (CollabLLM MediumDocEdit-Chat) and **coding** (CollabLLM
BigCodeBench-Chat) emit the CToM prompt variant directly. Both take the
downloaded parquet of each HF dataset's `train` split: the `-large` release is
the training set, and the items present only in the smaller release become the
held-out split.

```bash
hf download collabllm/collabllm-multiturn-medium-large        --repo-type dataset --local-dir src/medium-large
hf download collabllm/collabllm-multiturn-medium              --repo-type dataset --local-dir src/medium
hf download collabllm/collabllm-multiturn-bigcodebench-large  --repo-type dataset --local-dir src/bigcodebench-large
hf download collabllm/collabllm-multiturn-bigcodebench        --repo-type dataset --local-dir src/bigcodebench

python3 data_prep/build_writing.py \
    --large_file src/medium-large/data/train-00000-of-00001.parquet \
    --small_file src/medium/data/train-00000-of-00001.parquet \
    --local_dir data/collabllm-medium
```

Coding needs a **separate interpreter** for executing task code, pinned away from
the training environment (the task suites were authored against `numpy<2` /
`pandas 2.x`):

```bash
python3 -m venv bcb_venv
bcb_venv/bin/pip install -r data_prep/coding_sandbox_requirements.txt
export CTOM_BCB_PYTHON=$PWD/bcb_venv/bin/python
```

A task whose own reference solution cannot pass in that interpreter has a
constant-zero reward and a degenerate GRPO group, so the runnable set is
established first and the rest are dropped:

```bash
python3 data_prep/validate_coding_tasks.py \
    --files src/bigcodebench-large/data/*.parquet src/bigcodebench/data/*.parquet \
    --out data/bigcodebench_runnable.json

python3 data_prep/build_coding.py \
    --large_file src/bigcodebench-large/data/train-00000-of-00001.parquet \
    --small_file src/bigcodebench/data/train-00000-of-00001.parquet \
    --runnable data/bigcodebench_runnable.json \
    --local_dir data/collabllm-bigcodebench
```

Each processor writes `rl_{train,validation}.v7_prompt.parquet` alongside the
other prompt variants; the reported splits are 117 writing articles and 114
coding tasks. The coding runs in the paper read a file named
`.v7b_prompt.parquet`, which was an in-place re-stamp of an earlier split; a
fresh `data_prep/build_coding.py` run writes the identical prompt under the plain
`.v7_prompt` name, which is what `scripts/` expects.

Model-generated Python runs in short-lived, timed subprocesses with a denylist.
That matches the benchmark's execution model but is **not** a hardened sandbox —
run it only on disposable compute.

---

## 4. Train

### 4.1 Start the user simulator

One frozen `Qwen3-32B-FP8` serves the simulated user in every reported run;
personas come from the data, not from a second model.

```bash
bash scripts/serve_user_simulator.sh            # OpenAI-compatible, port 8100
```

For throughput, start replicas on several ports and pass the comma-separated
list to `--sim-url`.

### 4.2 Launch

```bash
# CToM
bash scripts/train_ctom.sh --domain math    --arm ctom
bash scripts/train_ctom.sh --domain writing --arm ctom
bash scripts/train_ctom.sh --domain coding  --arm ctom

# Ablations (paper Table 1): same controller and prompt, one or both training signals off
bash scripts/train_ctom.sh --domain math --arm no-effort       # CToM w/o effort      (w_effort = 0)
bash scripts/train_ctom.sh --domain math --arm no-prediction   # CToM w/o prediction  (lambda_pred = 0)
bash scripts/train_ctom.sh --domain math --arm task-only       # CToM task-only       (both off)

# Decision-entropy ablation
bash scripts/train_ctom.sh --domain math --arm ctom --decision-entropy 0.001

# Free-text controller instead of the categorical head
bash scripts/train_ctom.sh --domain math --arm ctom --action-protocol legacy
```

Useful flags: `--policy` (init, default `Qwen/Qwen3-8B`), `--sim-url`,
`--data-root`, `--save-dir`, `--nnodes`, `--ngpus-per-node`,
`--tensor-parallel`, `--inner-revisions`. Anything after `--` is forwarded to
Hydra verbatim, e.g. `-- trainer.total_epochs=1`.

The script writes the simulator interaction config, then calls
`verl.trainer.main_ppo` with the hyperparameters of the reported runs:

| | value |
| --- | --- |
| LoRA rank / alpha | 32 / 32, `all-linear` |
| Rollouts per prompt, prompt batch, PPO minibatch | 4, 36, 12 |
| Actor LR / schedule / warmup / min ratio / weight decay | 1e-4, cosine, 0.05, 0.1, 0.01 |
| Reference KL coefficient | 0.01 |
| Prediction-loss coefficient $\lambda_{\mathrm{pred}}$ | 0.1 (CToM, no-effort) / 0 (no-prediction, task-only) |
| Gradient-ratio cap $\gamma$ | 0.5 |
| Task / effort reward weight | 1.0 / 0.01 (CToM, no-prediction), 1.0 / 0 (no-effort, task-only) |
| Effort $\alpha$ / $\beta$ | 0.5 / 0.4 |
| Decision-entropy coefficient | 0 (main), 1e-3 (ablation) |
| Inner revision cap $K_{\max}$ | 3 |
| Advantage estimator | GRPO |
| Rollout / simulator temperature | 1.0 / 0.7 |
| Seed | 42 |

Per-domain settings — rounds, token budgets, cost budgets, epochs, and whether
the strict answer checker ends the episode — are in the `case "$DOMAIN"` block of
`scripts/train_ctom.sh`, one commented line each.

Each run writes `runs/<name>/v7_trace.jsonl`: every plan, decision, internal
draft, prediction and verdict, one file per rollout worker. Set
`CTOM_V7_TRACE=off` to disable it.

**Compute.** The reported runs used 8 single-GPU nodes: 4 GPUs for the policy and
4 serving the simulator pool. They ran 180 steps (math, 2 epochs over 3,263
problems) or 208 steps (writing and coding, 16 epochs over ~500 tasks), taking
roughly 16–24 h each. The defaults here (`--nnodes 1 --ngpus-per-node 8`) suit a
single 8-GPU host instead; pass `--nnodes`/`--ngpus-per-node` to match the
reported layout. `-- trainer.total_training_steps=3` gives a cheap end-to-end
smoke test of the whole loop.

---

## 5. Evaluate

`scripts/eval_ctom.sh` merges the LoRA adapter into the base model, serves it,
runs the domain's driver against the same frozen simulator, and scores.

```bash
bash scripts/eval_ctom.sh --domain math    --checkpoint runs/<run>/checkpoints/global_step_180
bash scripts/eval_ctom.sh --domain writing --checkpoint runs/<run>/checkpoints/global_step_208
bash scripts/eval_ctom.sh --domain coding  --checkpoint runs/<run>/checkpoints/global_step_208

# the untrained base anchor
bash scripts/eval_ctom.sh --domain math --checkpoint base
```

Pass `--action-protocol` and `--inner-revisions` matching the run being
evaluated. The evaluation controller is a mirror of the training controller, and
a mismatch does not error — it silently scores a different protocol.
`tests/test_eval_parity.py` exists to pin the two together.

Reported protocol, per domain:

| | math | writing | coding |
| --- | --- | --- | --- |
| Split | 365 held-out problems | 117 articles | 114 tasks |
| Rollouts per item | 8 (avg@8) | 1 | 8 (avg@8) |
| Assistant temperature | 1.0 | 1.0 | 1.0 |
| Simulator temperature | 0.7 | 0.7 | 0.7 |
| Rounds | 4 | 6 | 6 |
| Task metric | boxed-answer solve rate (avg@8) | ROUGE-L F1 | unit-test pass fraction (avg@8) |
| Scoring | inline | inline | inline |

Writing writes per-task rollouts and their ROUGE-L scores to
`rollouts.jsonl`; all three domains score inline, so no second stage is needed.

An errored rollout counts as a task failure; it is not dropped.

The prediction-agreement judge of Appendix B.5 / C.4 runs afterwards over the
math driver's `eval_results.jsonl`, and needs an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
python3 evaluation/score_prediction_judge.py --results eval_outputs/<label>/eval_results.jsonl
```

It reports the judge score alongside the lexical prediction F1 of Eq. (9), plus
the same-problem and cross-problem discrimination controls.


---

## License

This code is released under the [Apache License 2.0](LICENSE). The veRL patch in
`patches/` modifies [veRL](https://github.com/verl-project/verl), which is also
licensed under Apache 2.0.

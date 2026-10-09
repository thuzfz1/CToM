"""Every rollout must report the SAME reward_extra_info keys.

verl's agent_loop._postprocess reads the key list from rollout 0 and then
indexes every other rollout with it:

    reward_extra_keys = list(reward_extra_infos[0].keys())
    non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

So a key present in one rollout and absent in another is a hard KeyError in the
middle of generate_sequences, and WHICH rollout lands at index 0 decides whether
the step survives. That is how the collabllm-reward launches 948583 and 948961
died on a missing reward-extra key.

Run: .venv/bin/python -m pytest tests/test_reward_schema.py -q
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctom.reward import (  # noqa: E402
    REWARD_EXTRA_INFO_SCHEMA,
    _reward_extra_info,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(REPO, "ctom", "reward.py")).read()


def test_helper_always_returns_the_full_key_set():
    assert set(_reward_extra_info()) == set(REWARD_EXTRA_INFO_SCHEMA)
    partial = _reward_extra_info(**{"reward/total": 1.0})
    assert set(partial) == set(REWARD_EXTRA_INFO_SCHEMA)
    assert partial["reward/total"] == 1.0


def test_helper_rejects_keys_missing_from_the_schema():
    """An unknown key added at ONE call site is exactly the bug."""
    with pytest.raises(KeyError, match="REWARD_EXTRA_INFO_SCHEMA"):
        _reward_extra_info(**{"reward/brand_new": 1.0})


def test_every_reward_key_written_in_the_source_is_declared():
    """Catches `reward_extra_info["reward/x"] = ...` for an undeclared x."""
    written = set(re.findall(r'reward_extra_info\[\s*"([^"]+)"\s*\]\s*=', SRC))
    assert written, "expected the source to assign reward_extra_info keys"
    undeclared = sorted(written - set(REWARD_EXTRA_INFO_SCHEMA))
    assert not undeclared, f"assigned but not in the schema: {undeclared}"


def test_the_948961_crash_cannot_recur():
    """Reproduce verl's _postprocess over a batch that mixes a scored rollout
    with an empty one — the exact shape that killed 948583 and 948961."""
    scored = _reward_extra_info(**{
        "reward/task": 0.069, "reward/effort": 0.565,
        "reward/total": 0.335,
    })
    empty = _reward_extra_info(**{
        "reward/total": -1.0, "reward/empty_generation": True,
    })
    for first, second in ((scored, empty), (empty, scored)):
        keys = list(first.keys())          # verl takes rollout 0's keys ...
        for key in keys:
            second[key]                    # ... and indexes every other rollout


def test_no_return_path_builds_the_dict_by_hand():
    """Every `"reward_extra_info":` return site must go through the helper."""
    for m in re.finditer(r'"reward_extra_info":\s*(.{0,30})', SRC, re.S):
        tail = m.group(1).strip()
        assert tail.startswith("_reward_extra_info") or tail.startswith("reward_extra_info"), (
            f"hand-built reward_extra_info at ...{tail!r}; use _reward_extra_info()"
        )

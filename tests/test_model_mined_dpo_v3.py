from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from mv_audit.inference.batch_inference import _select_shard
from mv_audit.inference.sample_model_mined import _generate_batched
from mv_audit.training.reward_function import score_output
from mv_audit.training.train_dpo import _sequence_logps
from mv_audit.utils import iter_jsonl, read_yaml
from tools.compare_dpo_v3_results import _high_risk_misses
from tools.model_mined_dpo_v3 import _answer, _make_pair


V3_MIX = Path("docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix.jsonl")
SCHEMA = Path("configs/schema/output_schema.json")


def _order_row() -> dict:
    for row in iter_jsonl(V3_MIX):
        output = _answer(row)
        if "order_id_mismatch" in set(output.get("anomaly_types") or []):
            return row
    raise AssertionError("No order_id mismatch row found.")


def test_order_id_reward_requires_both_evidence_items() -> None:
    row = _order_row()
    truth = _answer(row)
    schema = read_yaml(SCHEMA)
    raw = json.dumps(truth, ensure_ascii=False)
    perfect = score_output(raw, {"output": truth}, row["images"], schema)
    assert perfect["details"]["r_order_id_pair"] == 1.0
    assert perfect["details"]["p_missing_order_id_pair"] == 0.0

    missing = copy.deepcopy(truth)
    missing["evidence"] = [
        item
        for item in missing["evidence"]
        if not (item.get("field") == "order_id" and item.get("source_doc_type") == "reimbursement_form")
    ]
    scored = score_output(json.dumps(missing, ensure_ascii=False), {"output": truth}, row["images"], schema)
    assert scored["details"]["p_missing_order_id_pair"] == 1.0
    assert scored["reward"] <= 0.0


def test_model_mined_pair_uses_real_ground_truth() -> None:
    row = _order_row()
    truth = _answer(row)
    rejected = copy.deepcopy(truth)
    rejected["risk_level"] = "low"
    rejected["audit_result"] = "pass"
    rejected["anomaly_types"] = []
    rejected["consistency_check"]["order_id_consistent"] = True
    rollout = {
        "case_id": row["case_id"],
        "prompt": row["messages"][0]["content"],
        "images": row["images"],
        "ground_truth": {"output": truth},
        "bucket": "order_id_mismatch",
        "completions": [
            {"raw_output": json.dumps(truth, ensure_ascii=False)},
            {"raw_output": json.dumps(rejected, ensure_ascii=False)},
        ],
    }
    schema = read_yaml(SCHEMA)
    pair = _make_pair(
        rollout,
        validator=Draft202012Validator(schema),
        schema=schema,
        min_gap=0.15,
        max_gap=2.0,
        allow_ground_truth_chosen=False,
    )
    assert pair is not None
    assert pair["chosen_source"] == "model_generated"
    assert pair["ground_truth"]["output"]["case_id"] == row["case_id"]
    assert pair["reward_gap"] >= 0.15


def _low_pass_row() -> dict:
    for row in iter_jsonl(V3_MIX):
        output = _answer(row)
        if output.get("risk_level") == "low" and output.get("audit_result") == "pass":
            return row
    raise AssertionError("No low/pass row found.")


def test_low_pass_pair_is_false_escalation_calibration() -> None:
    row = _low_pass_row()
    truth = _answer(row)
    rollout = {
        "case_id": row["case_id"],
        "prompt": row["messages"][0]["content"],
        "images": row["images"],
        "ground_truth": {"output": truth},
        "bucket": "low_pass",
        "completions": [{"raw_output": json.dumps(truth, ensure_ascii=False)}],
    }
    schema = read_yaml(SCHEMA)
    pair = _make_pair(
        rollout,
        validator=Draft202012Validator(schema),
        schema=schema,
        min_gap=0.15,
        max_gap=0.60,
        allow_ground_truth_chosen=False,
    )
    assert pair is not None
    assert pair["chosen_source"] == "model_generated"
    assert pair["rejected_source"] == "synthetic_false_escalation_calibration"
    rejected = json.loads(pair["rejected"])
    assert rejected["risk_level"] == "high"
    assert rejected["audit_result"] == "reject_recommendation"
    assert pair["chosen_task_reward"] > pair["rejected_task_reward"]


def test_order_model_error_can_use_expert_corrected_target() -> None:
    row = _order_row()
    truth = _answer(row)
    rejected = copy.deepcopy(truth)
    rejected["risk_level"] = "low"
    rejected["audit_result"] = "pass"
    rejected["anomaly_types"] = []
    rejected["consistency_check"]["order_id_consistent"] = True
    rollout = {
        "case_id": row["case_id"],
        "prompt": row["messages"][0]["content"],
        "images": row["images"],
        "ground_truth": {"output": truth},
        "bucket": "order_id_mismatch",
        "completions": [{"raw_output": json.dumps(rejected, ensure_ascii=False)}],
    }
    schema = read_yaml(SCHEMA)
    pair = _make_pair(
        rollout,
        validator=Draft202012Validator(schema),
        schema=schema,
        min_gap=0.15,
        max_gap=0.60,
        allow_ground_truth_chosen=True,
    )
    assert pair is not None
    assert pair["chosen_source"] == "expert_corrected_target"
    assert pair["rejected_source"] == "model_generated"
    assert pair["chosen_task_reward"] > pair["rejected_task_reward"]

def test_mean_token_logprob_normalization() -> None:
    if os.name == "nt":
        pytest.skip("Local Anaconda Torch aborts during MKL initialization on Windows; run on the Linux server.")
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __call__(self, **_: object) -> SimpleNamespace:
            logits = torch.tensor(
                [
                    [
                        [3.0, 1.0, 0.0],
                        [0.0, 3.0, 1.0],
                        [1.0, 0.0, 3.0],
                        [2.0, 1.0, 0.0],
                    ]
                ]
            )
            return SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.tensor([[0, 0, 0, 0]]),
        "labels": torch.tensor([[-100, 0, 1, 2]]),
    }
    summed = _sequence_logps(FakeModel(), batch, normalization="sum")
    averaged = _sequence_logps(FakeModel(), batch, normalization="mean_token")
    assert torch.allclose(averaged * 3, summed)


def test_shard_selection_is_complete_and_disjoint() -> None:
    rows = [{"case_id": str(index)} for index in range(17)]
    shards = [_select_shard(rows, shard_index=index, num_shards=5) for index in range(5)]
    flattened = [row["case_id"] for shard in shards for row in shard]
    assert sorted(flattened, key=int) == [str(index) for index in range(17)]
    assert len(flattened) == len(set(flattened))


def test_high_risk_miss_reader_supports_current_and_legacy_fields(tmp_path: Path) -> None:
    errors = tmp_path / "errors.jsonl"
    rows = [
        {"case_id": "current", "issues": ["high_risk_miss", "audit_mismatch"]},
        {"case_id": "legacy", "issue_codes": ["high_risk_miss"]},
        {"case_id": "other", "issues": ["bbox_strict_error"]},
    ]
    errors.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert _high_risk_misses(str(errors)) == {"current", "legacy"}

def test_batched_generation_returns_four_completions() -> None:
    import numpy as np

    class Inputs(dict):
        input_ids = np.zeros((1, 3), dtype=int)

    class Model:
        def generate(self, **kwargs):
            assert kwargs["num_return_sequences"] == 4
            return np.arange(28).reshape(4, 7)

    class Processor:
        def batch_decode(self, rows, **kwargs):
            assert rows.shape == (4, 4)
            return [f"completion-{index}" for index in range(4)]

    rows = _generate_batched(
        Model(),
        Processor(),
        Inputs(input_ids=Inputs.input_ids),
        count=4,
        max_new_tokens=16,
        temperature=0.8,
        top_p=0.95,
    )

    assert [row["raw_output"] for row in rows] == [f"completion-{index}" for index in range(4)]
    assert all(row["batched_generation"] is True for row in rows)

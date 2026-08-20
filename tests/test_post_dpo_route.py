import json
from types import SimpleNamespace

import pytest

from mv_audit.analysis.error_attribution import attribute_score
from mv_audit.analysis.post_dpo_route import assert_ready_for_rl, evaluate_repair_gate, evaluate_rl_decision


def _score(**overrides):
    data = {
        "case_id": "MV_MAIN_000001",
        "json_valid": True,
        "schema_valid": True,
        "field_score": 1.0,
        "anomaly_score": 1.0,
        "consistency_score": 1.0,
        "audit_correct": False,
        "risk_correct": True,
        "high_risk_miss": False,
        "false_manual_review": False,
        "false_escalation": False,
        "evidence_support": 1.0,
        "hallucination": 0.0,
        "bbox_score": 1.0,
        "truth_output": {"risk_level": "high", "audit_result": "reject_recommendation"},
        "pred_output": {"risk_level": "high", "audit_result": "manual_review"},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_error_attribution_marks_recognized_high_risk_release() -> None:
    attribution = attribute_score(_score())

    assert "decision_error" in attribution.error_tags
    assert attribution.decision_candidate is True
    assert attribution.problem_classes == ["recognized_risk_but_decision_released"]
    assert attribution.primary_problem_class == "recognized_risk_but_decision_released"


def test_error_attribution_marks_schema_and_high_risk_miss() -> None:
    attribution = attribute_score(
        _score(
            json_valid=False,
            schema_valid=False,
            high_risk_miss=True,
            pred_output=None,
            audit_correct=False,
            risk_correct=False,
        )
    )

    assert attribution.primary_error_source == "output_contract_error"
    assert attribution.primary_problem_class == "schema_contract_failure"
    assert "model_missed_high_risk" in attribution.problem_classes


def test_repair_main_gate_requires_high_risk_miss_improvement() -> None:
    config = {
        "repair_main_gate": {
            "json_validity_delta_min": -0.01,
            "schema_compliance_delta_min": -0.02,
            "audit_accuracy_delta_min": -0.01,
            "high_risk_miss_improvement_min": 0.03,
            "evidence_support_delta_min": -0.01,
        }
    }
    baseline = {
        "json_validity": 1.0,
        "schema_compliance": 0.876,
        "audit_accuracy": 0.773,
        "high_risk_miss_rate": 0.243,
        "evidence_support_rate": 0.804,
    }
    candidate = {
        "json_validity": 1.0,
        "schema_compliance": 0.870,
        "audit_accuracy": 0.770,
        "high_risk_miss_rate": 0.210,
        "evidence_support_rate": 0.800,
    }

    result = evaluate_repair_gate(
        baseline=baseline,
        candidate=candidate,
        config=config,
        gate_name="repair_main_gate",
    )

    assert result["passed"] is True
    assert result["checks"]["high_risk_miss_improvement"] is True


def test_rl_decision_requires_main_gate_and_decision_dominance() -> None:
    main_gate = {"passed": True}
    attribution_summary = {
        "residual_error_cases": 10,
        "decision_candidate_ratio": 0.7,
        "decision_primary_ratio": 0.6,
    }
    config = {
        "rl_decision": {
            "min_residual_error_cases": 5,
            "min_decision_candidate_ratio": 0.6,
            "min_decision_primary_ratio": 0.5,
        }
    }

    result = evaluate_rl_decision(main_gate=main_gate, attribution_summary=attribution_summary, config=config)

    assert result["status"] == "READY_FOR_RL"
    assert result["rl_recommended"] is True


def test_assert_ready_for_rl_rejects_unready_file(tmp_path) -> None:
    decision = tmp_path / "rl_decision.json"
    decision.write_text(json.dumps({"status": "RL_NOT_RECOMMENDED", "rl_recommended": False}), encoding="utf-8")

    with pytest.raises(SystemExit, match="RL is not ready"):
        assert_ready_for_rl(decision)
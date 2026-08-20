import json
from pathlib import Path


IDS = {
    "MV_MAIN_025856",
    "MV_MAIN_003978",
    "MV_MAIN_019752",
    "MV_MAIN_016083",
    "MV_MAIN_030493",
}

GROUND_TRUTH = Path("outputs/eval_sets/phase08_high_risk_repair_from_m2_train_decode_dev/train_decode_dev.jsonl")
PREDICTIONS = Path(
    "outputs/predictions/phase08_high_risk_repair_from_m2_train_decode_dev_schema_guarded/m2_sft/train_decode_dev.jsonl"
)


def load_truth():
    out = {}
    for line in GROUND_TRUTH.open(encoding="utf-8"):
        row = json.loads(line)
        case_id = row.get("case_id")
        if case_id in IDS:
            out[case_id] = row.get("answer") or row.get("output") or row
    return out


def load_predictions():
    out = {}
    for line in PREDICTIONS.open(encoding="utf-8"):
        row = json.loads(line)
        case_id = row.get("case_id")
        if case_id in IDS:
            out[case_id] = json.loads(row["raw_output"])
    return out


def false_checks(output):
    return [key for key, value in (output.get("consistency_check") or {}).items() if value is False]


def field_diffs(truth, pred):
    truth_fields = truth.get("field_extraction") or {}
    pred_fields = pred.get("field_extraction") or {}
    diffs = []
    for key in sorted(set(truth_fields) | set(pred_fields)):
        if truth_fields.get(key) != pred_fields.get(key):
            diffs.append(
                {
                    "field": key,
                    "truth": truth_fields.get(key),
                    "pred": pred_fields.get(key),
                }
            )
    return diffs


def classify(truth, pred):
    pred_anomalies = set(pred.get("anomaly_types") or [])
    pred_false_checks = set(false_checks(pred))
    pred_risk = pred.get("risk_level")
    pred_audit = pred.get("audit_result")
    if pred_risk == "high" and pred_audit == "pass":
        return "recognized_risk_but_decision_released"
    if pred_risk == "high" and pred_audit != truth.get("audit_result"):
        return "recognized_risk_but_wrong_decision"
    if pred_anomalies or pred_false_checks:
        return "recognized_some_signal_but_underestimated"
    return "model_missed_high_risk"


def main():
    truth = load_truth()
    pred = load_predictions()
    summaries = []
    for case_id in sorted(IDS):
        t = truth[case_id]
        p = pred[case_id]
        diffs = field_diffs(t, p)
        row = {
            "case_id": case_id,
            "classification": classify(t, p),
            "truth_risk_level": t.get("risk_level"),
            "truth_audit_result": t.get("audit_result"),
            "truth_anomaly_types": t.get("anomaly_types"),
            "truth_false_checks": false_checks(t),
            "pred_risk_level": p.get("risk_level"),
            "pred_audit_result": p.get("audit_result"),
            "pred_anomaly_types": p.get("anomaly_types"),
            "pred_false_checks": false_checks(p),
            "pred_reason": p.get("reason"),
            "field_diff_count": len(diffs),
            "field_diffs_preview": diffs[:8],
            "pred_evidence_fields": [item.get("field") for item in (p.get("evidence") or [])[:10]],
        }
        summaries.append(row)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

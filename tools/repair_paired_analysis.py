import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def case_id(row, index):
    for key in ("case_id", "id", "sample_id", "uid"):
        if key in row:
            return str(row[key])
    for key in ("meta", "metadata"):
        obj = row.get(key)
        if isinstance(obj, dict):
            for subkey in ("case_id", "id", "sample_id", "uid"):
                if subkey in obj:
                    return str(obj[subkey])
    return f"idx:{index}"


def answer_obj(row):
    raw = row.get("raw_output")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    for key in ("prediction", "parsed_prediction", "output", "response", "answer", "pred"):
        if key in row:
            return row[key]
    return row


def stable_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def get_path(obj, paths):
    for path in paths:
        cur = obj
        ok = True
        for part in path:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def predicted_decision(row):
    obj = answer_obj(row)
    return get_path(
        obj,
        [
            ("audit_result",),
            ("decision",),
            ("audit_decision",),
            ("risk_decision",),
            ("final_decision",),
            ("parsed", "audit_result"),
            ("parsed", "decision"),
            ("prediction", "audit_result"),
            ("prediction", "decision"),
        ],
    )


def predicted_risk_type(row):
    obj = answer_obj(row)
    return get_path(
        obj,
        [
            ("risk_type",),
            ("risk_level",),
            ("risk", "type"),
            ("risk", "risk_type"),
            ("parsed", "risk_type"),
            ("prediction", "risk_type"),
        ],
    )


def map_rows(rows):
    out = {}
    for idx, row in enumerate(rows):
        out[case_id(row, idx)] = row
    return out


def load_error_ids(path):
    rows = load_jsonl(path)
    ids = set()
    by_id = {}
    for idx, row in enumerate(rows):
        cid = case_id(row, idx)
        ids.add(cid)
        by_id[cid] = row
    return ids, by_id


def summarize_pair(name_a, rows_a, name_b, rows_b, error_ids_a=None, error_ids_b=None):
    a = map_rows(rows_a)
    b = map_rows(rows_b)
    common = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    same_full = 0
    same_answer = 0
    same_decision = 0
    same_risk = 0
    transitions = Counter()
    error_migration = Counter()
    changed_cases = []
    for cid in common:
        da = digest(a[cid])
        db = digest(b[cid])
        aa = answer_obj(a[cid])
        ab = answer_obj(b[cid])
        ans_same = digest(aa) == digest(ab)
        full_same = da == db
        dec_a = predicted_decision(a[cid])
        dec_b = predicted_decision(b[cid])
        risk_a = predicted_risk_type(a[cid])
        risk_b = predicted_risk_type(b[cid])
        same_full += int(full_same)
        same_answer += int(ans_same)
        same_decision += int(dec_a == dec_b)
        same_risk += int(risk_a == risk_b)
        transitions[(str(dec_a), str(dec_b))] += 1
        if error_ids_a is not None and error_ids_b is not None:
            ea = cid in error_ids_a
            eb = cid in error_ids_b
            if ea and not eb:
                key = f"{name_a}_wrong_to_{name_b}_right"
            elif not ea and eb:
                key = f"{name_a}_right_to_{name_b}_wrong"
            elif ea and eb:
                key = "both_wrong"
            else:
                key = "both_right"
            error_migration[key] += 1
        if not ans_same:
            changed_cases.append(
                {
                    "case_id": cid,
                    f"{name_a}_decision": dec_a,
                    f"{name_b}_decision": dec_b,
                    f"{name_a}_risk_type": risk_a,
                    f"{name_b}_risk_type": risk_b,
                    f"{name_a}_answer_sha256": digest(aa),
                    f"{name_b}_answer_sha256": digest(ab),
                }
            )
    summary = {
        "pair": f"{name_a}_vs_{name_b}",
        "common_cases": len(common),
        f"only_{name_a}": len(only_a),
        f"only_{name_b}": len(only_b),
        "same_full_rows": same_full,
        "same_answer_objects": same_answer,
        "same_answer_rate": same_answer / len(common) if common else None,
        "changed_answer_objects": len(common) - same_answer,
        "same_decision_rate": same_decision / len(common) if common else None,
        "same_risk_type_rate": same_risk / len(common) if common else None,
        "decision_transitions": {
            f"{k[0]} -> {k[1]}": v for k, v in transitions.most_common()
        },
        "error_migration": dict(error_migration),
        "changed_cases_preview": changed_cases[:30],
    }
    return summary, changed_cases


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)




def summarize_errors(error_path, transition_csv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    error_rows = load_jsonl(error_path)
    issue_counter = Counter()
    class_counter = Counter()
    risk_counter = Counter()
    audit_counter = Counter()
    examples = defaultdict(list)

    def classes_for_issues(row):
        issues = set(row.get("issues") or [])
        classes = []
        if "json_invalid" in issues or "schema_invalid" in issues or "business_metrics_zeroed" in issues:
            classes.append("schema_contract_failure")
        pred_risk = row.get("pred_risk_level")
        pred_audit = row.get("pred_audit_result")
        truth_risk = row.get("truth_risk_level")
        truth_audit = row.get("truth_audit_result")
        if truth_risk == "high" and (pred_risk != "high" or pred_audit == "pass"):
            classes.append("model_missed_high_risk")
        if truth_risk == "high" and pred_risk == "high" and truth_audit != pred_audit:
            classes.append("recognized_risk_but_decision_released")
        if "unsupported_evidence" in issues or "bbox_strict_error" in issues or "hallucination" in issues:
            classes.append("evidence_not_trustworthy")
        if "audit_mismatch" in issues or "risk_mismatch" in issues:
            classes.append("decision_or_risk_mismatch")
        return classes or ["other"]

    for row in error_rows:
        cid = str(row.get("case_id"))
        for issue in row.get("issues") or []:
            issue_counter[issue] += 1
        risk_counter[f"{row.get('truth_risk_level')} -> {row.get('pred_risk_level')}"] += 1
        audit_counter[f"{row.get('truth_audit_result')} -> {row.get('pred_audit_result')}"] += 1
        for cls in classes_for_issues(row):
            class_counter[cls] += 1
            if len(examples[cls]) < 10:
                examples[cls].append(cid)

    transition_issue_counter = Counter()
    transition_counter = Counter()
    if transition_csv:
        with Path(transition_csv).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transition_counter[row.get("transition") or ""] += 1
                for issue in (row.get("candidate_issues") or "").split("|"):
                    if issue:
                        transition_issue_counter[issue] += 1

    summary = {
        "error_file": str(error_path),
        "error_rows": len(error_rows),
        "issue_counts": dict(issue_counter.most_common()),
        "problem_class_counts": dict(class_counter.most_common()),
        "truth_to_pred_risk_counts": dict(risk_counter.most_common()),
        "truth_to_pred_audit_counts": dict(audit_counter.most_common()),
        "examples_by_problem_class": dict(examples),
        "transition_counts": dict(transition_counter.most_common()),
        "transition_candidate_issue_counts": dict(transition_issue_counter.most_common()),
    }
    (out / "repair_error_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-pred", required=True)
    parser.add_argument("--dpo-pred", required=True)
    parser.add_argument("--dpo2-pred")
    parser.add_argument("--repair-errors", required=True)
    parser.add_argument("--dpo-errors", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    repair_rows = load_jsonl(args.repair_pred)
    dpo_rows = load_jsonl(args.dpo_pred)
    repair_error_ids, _ = load_error_ids(args.repair_errors)
    dpo_error_ids, _ = load_error_ids(args.dpo_errors)

    summaries = []
    changed_all = []
    summary, changed = summarize_pair(
        "repair_from_m2",
        repair_rows,
        "dpo_v2",
        dpo_rows,
        repair_error_ids,
        dpo_error_ids,
    )
    summaries.append(summary)
    changed_all.extend(changed)

    if args.dpo2_pred:
        dpo2_rows = load_jsonl(args.dpo2_pred)
        summary, changed = summarize_pair(
            "repair_from_m2",
            repair_rows,
            "dpo_v2_baseline_ablation",
            dpo2_rows,
        )
        summaries.append(summary)
        changed_all.extend(changed)

    (out / "paired_diff_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out / "paired_changed_cases.csv", changed_all)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if "--summarize-errors" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--summarize-errors", required=True)
        p.add_argument("--transition-csv")
        p.add_argument("--out-dir", required=True)
        ns = p.parse_args()
        summarize_errors(ns.summarize_errors, ns.transition_csv, ns.out_dir)
    else:
        main()
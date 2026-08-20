"""Create a runtime inference config for one DPO v3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from mv_audit.utils import read_yaml, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/train/dpo_v3_model_mined_qwen3vl_8b_server.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--ground_truth_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = read_yaml(args.base)
    inference = config.setdefault("inference", {})
    inference["dpo_adapter_dir"] = args.adapter
    inference["predictions_dir"] = args.predictions_dir
    inference["ground_truth_dir"] = args.ground_truth_dir
    inference["train_decode_dev_ground_truth_dir"] = args.ground_truth_dir
    write_yaml(config, args.output)
    print(Path(args.output))


if __name__ == "__main__":
    main()
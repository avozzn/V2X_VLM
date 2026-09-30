#!/usr/bin/env python3
"""Run the corrected V2X planning and collision evaluation from a prediction PKL."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.evaluation import planning_evaluation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trajectory L2 and collision rate with corrected V2X "
            "coordinate orientation and occupancy time alignment."
        )
    )
    parser.add_argument(
        "prediction",
        type=Path,
        help="Prediction PKL containing a dict: token -> trajectory coordinates.",
    )
    parser.add_argument(
        "--predict-steps",
        type=int,
        choices=(6, 9, 12),
        default=9,
        help=(
            "Number of 0.5-second future steps to evaluate (default: 9). "
            "The current V2X occupancy GT supports at most 10 future steps."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results",
        help="Directory for summary and per-token analysis JSON files.",
    )
    return parser.parse_args()


def load_predictions(path):
    with path.open("rb") as file:
        predictions = pickle.load(file)
    if not isinstance(predictions, dict):
        raise TypeError(
            f"Prediction file must contain a dict, got {type(predictions).__name__}."
        )
    if not predictions:
        raise ValueError("Prediction dictionary is empty.")
    return predictions


def main():
    args = parse_args()
    prediction_path = args.prediction.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(prediction_path)
    run_name = f"{prediction_path.stem}_{args.predict_steps}steps"
    summary_path = output_dir / f"{run_name}_metrics.json"
    analysis_path = output_dir / f"{run_name}_analysis.json"

    print(f"Prediction: {prediction_path}")
    print(f"Tokens: {len(predictions)}")
    print(f"Horizon: {args.predict_steps} steps ({args.predict_steps * 0.5:.1f}s)")

    # planning_evaluation writes its per-token dump to the current directory.
    previous_cwd = Path.cwd()
    temporary_analysis_path = output_dir / "evaluation_analysis_dump.json"
    try:
        os.chdir(output_dir)
        metrics = planning_evaluation(
            predictions,
            subset=None,
            only_vehicle=False,
            predict_steps=args.predict_steps,
        )
    finally:
        os.chdir(previous_cwd)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    if temporary_analysis_path.exists():
        temporary_analysis_path.replace(analysis_path)

    print(f"\nMetrics saved to: {summary_path}")
    if analysis_path.exists():
        print(f"Per-token analysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()

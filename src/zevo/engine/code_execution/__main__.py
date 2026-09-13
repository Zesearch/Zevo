from __future__ import annotations

import argparse
import json
from pathlib import Path

from zevo.engine.code_execution.scorer import score_code_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a registered coding benchmark")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scoring-set", required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    metrics = score_code_benchmark(
        adapter=args.adapter,
        predictions=Path(args.predictions),
        scoring_set=Path(args.scoring_set),
        prediction_column=args.prediction_column,
    )
    Path(args.metrics_out).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if metrics.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

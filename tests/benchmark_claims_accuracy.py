import argparse

from benchmark_claims_individual_common import (
    add_common_args,
    build_cfg,
    run_rows,
    write_filtered_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy benchmark (MAE, threshold metrics, retrieval metrics)."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = build_cfg(args, include_scaling=False)
    rows, skipped = run_rows(cfg, include_near_duplicates=True)
    write_filtered_outputs(
        cfg=cfg,
        test_name="accuracy",
        rows=rows,
        skipped=skipped,
        keep_metric_roots=(
            "mae_",
            "precision_threshold_",
            "recall_threshold_",
            "f1_threshold_",
            "precision_at_k_",
            "recall_at_k_",
        ),
    )


if __name__ == "__main__":
    main()

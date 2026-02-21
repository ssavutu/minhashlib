import argparse

from benchmark_claims_individual_common import (
    add_common_args,
    build_cfg,
    run_scaling_rows,
    write_filtered_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Scaling benchmark (vary docs/num_perm/doc length on synthetic data)."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = build_cfg(args, include_scaling=True)
    rows = run_scaling_rows(cfg)
    write_filtered_outputs(
        cfg=cfg,
        test_name="scaling",
        rows=rows,
        skipped=[],
        keep_metric_roots=(
            "build_seconds_",
            "pair_eval_seconds_",
            "retrieval_seconds_",
            "docs_per_second_",
            "mae_",
            "peak_alloc_bytes_",
            "bytes_per_signature_",
        ),
    )


if __name__ == "__main__":
    main()

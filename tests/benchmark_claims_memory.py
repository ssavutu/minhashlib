import argparse

from benchmark_claims_individual_common import (
    add_common_args,
    build_cfg,
    run_rows,
    write_filtered_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Memory benchmark (peak allocation and bytes/signature)."
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = build_cfg(args, include_scaling=False)
    rows, skipped = run_rows(cfg, include_near_duplicates=True)
    write_filtered_outputs(
        cfg=cfg,
        test_name="memory",
        rows=rows,
        skipped=skipped,
        keep_metric_roots=(
            "peak_alloc_bytes_",
            "bytes_per_signature_",
        ),
    )


if __name__ == "__main__":
    main()

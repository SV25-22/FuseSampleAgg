from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DGL_VARIANTS = ["dgl_gpu", "dgl_cpu_workers", "dgl_cpu_uva"]


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def node_table(path: Path) -> pd.DataFrame:
    frame = numeric(
        pd.read_csv(path),
        ["batch_size", "median_step_s", "transient_peak_mb", "seed"],
    )
    required = {"dataset", "variant", "fanout", "batch_size", "median_step_s"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} does not have the node benchmark schema")
    grouped = frame.groupby(["dataset", "fanout", "batch_size", "variant"], as_index=False).agg(
        runs=("seed", "count"),
        step_ms=("median_step_s", lambda values: 1000 * np.nanmedian(values)),
        transient_peak_mb=("transient_peak_mb", "median"),
    )
    rows = []
    for config, subset in grouped.groupby(["dataset", "fanout", "batch_size"]):
        fsa = subset[subset.variant == "fsa"]
        dgl = subset[subset.variant.isin(DGL_VARIANTS)]
        if fsa.empty or dgl.empty:
            continue
        fsa = fsa.iloc[0]
        dgl = dgl.sort_values("step_ms").iloc[0]
        rows.append(
            {
                "dataset": config[0],
                "fanout": config[1],
                "batch_size": config[2],
                "best_dgl": dgl.variant,
                "dgl_step_ms": dgl.step_ms,
                "fsa_step_ms": fsa.step_ms,
                "speedup": dgl.step_ms / fsa.step_ms,
                "dgl_transient_peak_mb": dgl.transient_peak_mb,
                "fsa_transient_peak_mb": fsa.transient_peak_mb,
                "runs": min(int(dgl.runs), int(fsa.runs)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["dataset", "fanout", "batch_size"])


def parity_table(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        final = frame.iloc[-1]
        rows.append(final.to_dict())
    if not rows:
        return pd.DataFrame()
    frame = numeric(
        pd.DataFrame(rows),
        ["training_wall_s", "train_acc", "valid_acc", "test_acc", "seed"],
    )
    return (
        frame.groupby(["dataset", "variant"], as_index=False)
        .agg(
            runs=("seed", "count"),
            training_wall_mean_s=("training_wall_s", "mean"),
            training_wall_std_s=("training_wall_s", "std"),
            train_acc_mean=("train_acc", "mean"),
            train_acc_std=("train_acc", "std"),
            valid_acc_mean=("valid_acc", "mean"),
            valid_acc_std=("valid_acc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
        )
        .sort_values(["dataset", "variant"])
    )


def kg_table(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if {"dataset", "variant", "median_step_ms"}.issubset(record):
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    frame = numeric(
        pd.DataFrame(rows),
        [
            "batch_size",
            "steps",
            "seed",
            "median_step_ms",
            "peak_vram_mb",
            "transient_peak_mb",
            "valid_mrr",
            "test_mrr",
        ],
    )
    for column in ["valid_mrr", "test_mrr", "transient_peak_mb"]:
        if column not in frame:
            frame[column] = np.nan
    summary = frame.groupby(
        ["dataset", "fanout", "batch_size", "steps", "variant"],
        as_index=False,
    ).agg(
        runs=("seed", "count"),
        step_mean_ms=("median_step_ms", "mean"),
        step_std_ms=("median_step_ms", "std"),
        peak_mean_mb=("peak_vram_mb", "mean"),
        peak_std_mb=("peak_vram_mb", "std"),
        transient_peak_mean_mb=("transient_peak_mb", "mean"),
        valid_mrr_mean=("valid_mrr", "mean"),
        valid_mrr_std=("valid_mrr", "std"),
        test_mrr_mean=("test_mrr", "mean"),
        test_mrr_std=("test_mrr", "std"),
    )
    rows = []
    for config, subset in summary.groupby(["dataset", "fanout", "batch_size", "steps"]):
        dgl = subset[subset.variant == "dgl"]
        fsa = subset[subset.variant == "fsa"]
        if dgl.empty or fsa.empty:
            continue
        dgl = dgl.iloc[0]
        fsa = fsa.iloc[0]
        rows.append(
            {
                "dataset": config[0],
                "fanout": config[1],
                "batch_size": config[2],
                "steps": config[3],
                "runs": min(int(dgl.runs), int(fsa.runs)),
                "dgl_step_mean_ms": dgl.step_mean_ms,
                "dgl_step_std_ms": dgl.step_std_ms,
                "fsa_step_mean_ms": fsa.step_mean_ms,
                "fsa_step_std_ms": fsa.step_std_ms,
                "speedup": dgl.step_mean_ms / fsa.step_mean_ms,
                "dgl_peak_mean_mb": dgl.peak_mean_mb,
                "fsa_peak_mean_mb": fsa.peak_mean_mb,
                "dgl_valid_mrr_mean": dgl.valid_mrr_mean,
                "fsa_valid_mrr_mean": fsa.valid_mrr_mean,
                "dgl_test_mrr_mean": dgl.test_mrr_mean,
                "fsa_test_mrr_mean": fsa.test_mrr_mean,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["dataset", "fanout", "batch_size", "steps"])


def write_table(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".tex").write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/tables"))
    args = parser.parse_args()

    node_path = args.results / "node.csv"
    if node_path.exists():
        write_table(node_table(node_path), args.out / "node_performance")
    write_table(parity_table(args.results / "parity"), args.out / "accuracy_parity")
    write_table(kg_table(args.results / "kg"), args.out / "kg_completion")


if __name__ == "__main__":
    main()

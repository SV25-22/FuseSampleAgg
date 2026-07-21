from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["dgl_gpu", "dgl_cpu_workers", "dgl_cpu_uva", "fsa"]
FIELDS = [
    "timestamp",
    "dataset",
    "variant",
    "fanout",
    "batch_size",
    "hidden",
    "precision",
    "steps",
    "warmup",
    "seed",
    "repeat",
    "median_step_s",
    "ci_lo_s",
    "ci_hi_s",
    "transient_peak_mb",
]


def run_configuration(arguments: list[str]) -> dict:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"benchmark produced no JSON record: {' '.join(arguments)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["reddit", "ogbn-arxiv", "ogbn-products"],
    )
    parser.add_argument("--fanouts", nargs="+", default=["15 10", "25 10"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "node.csv")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in args.datasets:
            csr_path = args.data_root / dataset / "csr_adj.npz"
            if not csr_path.exists():
                raise FileNotFoundError(f"missing {csr_path}; run scripts/download_data.py first")
            for fanout in args.fanouts:
                fanout_values = fanout.split()
                if len(fanout_values) != 2:
                    raise ValueError(f"expected two fanouts, received {fanout!r}")
                for batch_size in args.batch_sizes:
                    for repeat in range(args.repeats):
                        seed = args.seed + repeat
                        for variant in VARIANTS:
                            command = [
                                sys.executable,
                                "-m",
                                "train.bench_fsa_vs_dgl",
                                "--variant",
                                variant,
                                "--dataset",
                                dataset,
                                "--data-root",
                                str(args.data_root),
                                "--fanout",
                                *fanout_values,
                                "--batch-size",
                                str(batch_size),
                                "--hidden",
                                str(args.hidden),
                                "--steps",
                                str(args.steps),
                                "--warmup",
                                str(args.warmup),
                                "--seed",
                                str(seed),
                            ]
                            record = run_configuration(command)
                            record["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
                            record["repeat"] = repeat
                            writer.writerow({field: record.get(field, "") for field in FIELDS})
                            handle.flush()
                            print(
                                dataset,
                                fanout,
                                batch_size,
                                repeat,
                                variant,
                                f"{record['median_step_s'] * 1000:.3f} ms",
                            )


if __name__ == "__main__":
    main()

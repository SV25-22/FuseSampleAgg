from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"no JSON result from {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["performance", "quality"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "kg")
    args = parser.parse_args()

    if args.mode == "performance":
        datasets = ["ogbl-wikikg2", "ogbl-biokg"]
        fanouts = [(15, 10), (25, 10)]
        batch_sizes = [512, 1024]
        steps = 200
        evaluate = False
    else:
        datasets = ["ogbl-wikikg2"]
        fanouts = [(15, 10), (25, 10)]
        batch_sizes = [1024]
        steps = 6000
        evaluate = True

    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        if not (args.data_root / dataset / "csr_adj.npz").exists():
            raise FileNotFoundError(f"prepare {dataset} with scripts/kg_prepare_csr.py first")
        for fanout1, fanout2 in fanouts:
            for batch_size in batch_sizes:
                for seed in range(42, 47):
                    for variant in ("dgl", "fsa"):
                        tag = "quality" if evaluate else "performance"
                        output = args.out / (
                            f"{tag}_{dataset}_{variant}_f{fanout1}-{fanout2}_"
                            f"b{batch_size}_s{seed}.json"
                        )
                        command = [
                            sys.executable,
                            "-m",
                            "train.kg_completion_bench",
                            "--variant",
                            variant,
                            "--dataset",
                            dataset,
                            "--data-root",
                            str(args.data_root),
                            "--fanout",
                            str(fanout1),
                            str(fanout2),
                            "--batch-size",
                            str(batch_size),
                            "--steps",
                            str(steps),
                            "--warmup",
                            "20",
                            "--seed",
                            str(seed),
                            "--json-out",
                            str(output),
                        ]
                        if evaluate:
                            command.append("--evaluate")
                        result = run(command)
                        print(
                            dataset,
                            variant,
                            fanout1,
                            fanout2,
                            batch_size,
                            seed,
                            f"{result['median_step_ms']:.3f} ms",
                        )


if __name__ == "__main__":
    main()

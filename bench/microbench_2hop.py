from __future__ import annotations

import argparse
import json

import torch

from fuseop import fused_sample_agg_2hop


def regular_csr(num_nodes: int, degree: int):
    rowptr = torch.arange(0, (num_nodes + 1) * degree, degree, dtype=torch.int32)
    col = torch.randint(0, num_nodes, (num_nodes * degree,), dtype=torch.int32)
    return rowptr.cuda(), col.cuda()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--degree", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fanout", nargs=2, type=int, default=[15, 10])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    rowptr, col = regular_csr(args.nodes, args.degree)
    features = torch.randn(args.nodes, args.features, device="cuda")
    roots = torch.randint(0, args.nodes, (args.batch_size,), dtype=torch.int32, device="cuda")
    fanout1, fanout2 = args.fanout

    for step in range(args.warmup):
        fused_sample_agg_2hop(rowptr, col, features, roots, fanout1, fanout2, args.seed + step)
    torch.cuda.synchronize()

    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for step in range(args.steps):
        fused_sample_agg_2hop(
            rowptr,
            col,
            features,
            roots,
            fanout1,
            fanout2,
            args.seed + args.warmup + step,
        )
    finished.record()
    finished.synchronize()
    total_ms = started.elapsed_time(finished)
    sampled_edges = args.batch_size * fanout1 * fanout2 * args.steps
    print(
        json.dumps(
            {
                "mean_step_ms": total_ms / args.steps,
                "sampled_edges_per_second": sampled_edges / (total_ms / 1000),
                "steps": args.steps,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

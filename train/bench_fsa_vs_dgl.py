from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import dgl
import dgl.function as dgl_fn
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fuseop import fused_sample_agg_2hop


class MeanHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.neighbor_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, self_features: torch.Tensor, neighbor_mean: torch.Tensor):
        hidden = self.self_projection(self_features)
        hidden = hidden + self.neighbor_projection(neighbor_mean)
        return self.output(F.relu(hidden))


def load_dataset(name: str, root: Path):
    if name == "reddit":
        from torch_geometric.datasets import Reddit

        dataset = Reddit(root=str(root))
        data = dataset[0]
        labels = data.y.view(-1).long()
        train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
        num_classes = int(labels.max().item()) + 1
    else:
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name=name, root=str(root))
        data = dataset[0]
        labels = data.y.view(-1).long()
        train_nodes = dataset.get_idx_split()["train"]
        num_classes = int(dataset.num_classes)
    return data.x.float(), labels, train_nodes, num_classes


def load_csr(path: Path):
    with np.load(path) as data:
        rowptr = torch.from_numpy(data["rowptr"].astype(np.int32, copy=False))
        col = torch.from_numpy(data["col"].astype(np.int32, copy=False))
    return rowptr.contiguous(), col.contiguous()


def csr_rows(rowptr: torch.Tensor) -> torch.Tensor:
    degrees = (rowptr[1:] - rowptr[:-1]).to(torch.int64)
    return torch.arange(rowptr.numel() - 1, dtype=torch.int32).repeat_interleave(degrees)


def build_dgl_graph(rowptr: torch.Tensor, col: torch.Tensor, mode: str) -> dgl.DGLGraph:
    rows = csr_rows(rowptr)
    device = "cuda" if mode == "dgl_gpu" else "cpu"
    rows = rows.to(device)
    neighbors = col.to(device)
    graph = dgl.graph(
        (neighbors, rows),
        num_nodes=rowptr.numel() - 1,
        idtype=torch.int32,
        device=device,
    )
    if mode == "dgl_cpu_uva":
        graph.pin_memory_()
    return graph


def build_dgl_loader(
    graph: dgl.DGLGraph,
    train_nodes: torch.Tensor,
    fanout1: int,
    fanout2: int,
    batch_size: int,
    mode: str,
):
    sampler = dgl.dataloading.NeighborSampler([fanout2, fanout1])
    if mode == "dgl_gpu":
        seeds = train_nodes.to("cuda", dtype=torch.int32)
        workers = 0
        use_uva = False
        prefetch = False
    elif mode == "dgl_cpu_uva":
        seeds = train_nodes.to("cuda", dtype=torch.int32)
        workers = 0
        use_uva = True
        prefetch = False
    else:
        seeds = train_nodes.to("cpu", dtype=torch.int32)
        workers = 8
        use_uva = False
        prefetch = True
    return dgl.dataloading.DataLoader(
        graph,
        seeds,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        device="cuda",
        num_workers=workers,
        use_uva=use_uva,
        use_prefetch_thread=prefetch,
    )


class CyclingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def repeat_seed_batches(
    train_nodes: torch.Tensor, batch_size: int, seed: int
) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    nodes = train_nodes.to("cuda", dtype=torch.int32)
    while True:
        order = torch.randperm(nodes.numel(), generator=generator, device="cuda")
        usable = nodes.numel() - nodes.numel() % batch_size
        for start in range(0, usable, batch_size):
            yield nodes[order[start : start + batch_size]]


def nested_mean_from_blocks(blocks, features: torch.Tensor) -> torch.Tensor:
    farthest, closest = blocks
    source_ids = farthest.srcdata[dgl.NID].long()
    middle = dgl.ops.copy_u_mean(farthest, features[source_ids])
    valid = farthest.in_degrees().gt(0).to(middle.dtype).unsqueeze(1)
    closest.srcdata["middle"] = middle
    closest.srcdata["valid"] = valid
    closest.update_all(dgl_fn.copy_u("middle", "message"), dgl_fn.sum("message", "sum"))
    closest.update_all(dgl_fn.copy_u("valid", "message"), dgl_fn.sum("message", "count"))
    return closest.dstdata["sum"] / closest.dstdata["count"].clamp_min(1)


def dgl_step(
    model: MeanHead,
    batch,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    _, output_nodes, blocks = batch
    roots = output_nodes.long()
    neighbor_mean = nested_mean_from_blocks(blocks, features)
    logits = model(features[roots], neighbor_mean)
    loss = F.cross_entropy(logits, labels[roots])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fsa_step(
    model: MeanHead,
    roots: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    fanout1: int,
    fanout2: int,
    replay_seed: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    neighbor_mean = fused_sample_agg_2hop(
        rowptr,
        col,
        features,
        roots,
        fanout1,
        fanout2,
        replay_seed=replay_seed,
    )
    root_ids = roots.long()
    logits = model(features[root_ids], neighbor_mean)
    loss = F.cross_entropy(logits, labels[root_ids])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def median_interval(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    lower = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return median, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=["fsa", "dgl_gpu", "dgl_cpu_workers", "dgl_cpu_uva"],
    )
    parser.add_argument(
        "--dataset", required=True, choices=["reddit", "ogbn-arxiv", "ogbn-products"]
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--fanout", nargs=2, type=int, default=[15, 10])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset_root = args.data_root / args.dataset
    features_cpu, labels_cpu, train_nodes, num_classes = load_dataset(args.dataset, dataset_root)
    rowptr_cpu, col_cpu = load_csr(dataset_root / "csr_adj.npz")
    features = features_cpu.to("cuda").contiguous()
    labels = labels_cpu.to("cuda")

    torch.manual_seed(args.seed)
    model = MeanHead(features.size(1), args.hidden, num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    fanout1, fanout2 = args.fanout

    if args.variant == "fsa":
        rowptr = rowptr_cpu.cuda().contiguous()
        col = col_cpu.cuda().contiguous()
        batches = repeat_seed_batches(train_nodes, args.batch_size, args.seed)

        def run(step: int):
            roots = next(batches)
            return fsa_step(
                model,
                roots,
                rowptr,
                col,
                features,
                labels,
                fanout1,
                fanout2,
                args.seed * 1_000_003 + step,
                optimizer,
            )

    else:
        graph = build_dgl_graph(rowptr_cpu, col_cpu, args.variant)
        loader = build_dgl_loader(
            graph,
            train_nodes,
            fanout1,
            fanout2,
            args.batch_size,
            args.variant,
        )
        batches = CyclingLoader(loader)

        def run(step: int):
            dgl.seed(args.seed * 1_000_003 + step)
            return dgl_step(model, next(batches), features, labels, optimizer)

    for step in range(args.warmup):
        run(step)
    torch.cuda.synchronize()
    gc.collect()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        run(args.warmup + step)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)

    median, lower, upper = median_interval(timings)
    transient_peak = max(0, torch.cuda.max_memory_allocated() - baseline_bytes)
    result = {
        "variant": args.variant,
        "dataset": args.dataset,
        "fanout": f"{fanout1} {fanout2}",
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "precision": "fp32",
        "steps": len(timings),
        "warmup": args.warmup,
        "seed": args.seed,
        "median_step_s": median,
        "ci_lo_s": lower,
        "ci_hi_s": upper,
        "transient_peak_mb": transient_peak / (1024**2),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

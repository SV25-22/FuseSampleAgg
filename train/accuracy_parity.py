from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import scatter

from fuseop import fused_sample_agg_2hop


class ParityHead(nn.Module):
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
        data.y = data.y.view(-1).long()
        split = {
            "train": data.train_mask.nonzero(as_tuple=False).view(-1),
            "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
            "test": data.test_mask.nonzero(as_tuple=False).view(-1),
        }
        num_classes = int(data.y.max().item()) + 1
    else:
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name=name, root=str(root))
        data = dataset[0]
        data.y = data.y.view(-1).long()
        split = dataset.get_idx_split()
        num_classes = int(dataset.num_classes)
    return data, split, num_classes


def load_csr(path: Path):
    with np.load(path) as data:
        rowptr = torch.from_numpy(data["rowptr"].astype(np.int32, copy=False))
        col = torch.from_numpy(data["col"].astype(np.int32, copy=False))
    return rowptr.cuda().contiguous(), col.cuda().contiguous()


def pyg_nested_mean(batch) -> torch.Tensor:
    sampled_edges = [int(value) for value in batch.num_sampled_edges]
    if len(sampled_edges) != 2:
        raise RuntimeError("the parity runner requires two sampled hops")
    first_count, second_count = sampled_edges
    first = batch.edge_index[:, :first_count]
    second = batch.edge_index[:, first_count : first_count + second_count]
    inner_sum = scatter(
        batch.x[second[0]],
        second[1],
        dim=0,
        dim_size=batch.num_nodes,
        reduce="sum",
    )
    inner_count = scatter(
        torch.ones_like(second[1], dtype=batch.x.dtype),
        second[1],
        dim=0,
        dim_size=batch.num_nodes,
        reduce="sum",
    )
    inner = inner_sum / inner_count.clamp_min(1).unsqueeze(1)
    first_valid = inner_count[first[0]].gt(0).to(batch.x.dtype)
    outer_sum = scatter(
        inner[first[0]] * first_valid.unsqueeze(1),
        first[1],
        dim=0,
        dim_size=batch.num_nodes,
        reduce="sum",
    )
    outer_count = scatter(
        first_valid,
        first[1],
        dim=0,
        dim_size=batch.num_nodes,
        reduce="sum",
    )
    outer = outer_sum / outer_count.clamp_min(1).unsqueeze(1)
    return outer[: batch.batch_size]


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_pyg(model, data, split, fanouts, batch_size) -> dict[str, float]:
    model.eval()
    results = {}
    for split_name, nodes in split.items():
        loader = NeighborLoader(
            data,
            num_neighbors=fanouts,
            input_nodes=nodes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        correct = 0
        total = 0
        for batch in loader:
            batch = batch.cuda()
            logits = model(batch.x[: batch.batch_size], pyg_nested_mean(batch))
            target = batch.y[: batch.batch_size]
            correct += int((logits.argmax(dim=-1) == target).sum())
            total += int(target.numel())
        results[split_name] = accuracy(correct, total)
    return results


@torch.no_grad()
def evaluate_fsa(
    model,
    features,
    labels,
    split,
    rowptr,
    col,
    fanout1,
    fanout2,
    batch_size,
    seed,
) -> dict[str, float]:
    model.eval()
    results = {}
    for split_offset, (split_name, nodes) in enumerate(split.items()):
        correct = 0
        total = 0
        for start in range(0, nodes.numel(), batch_size):
            roots = nodes[start : start + batch_size].cuda().to(torch.int32)
            mean = fused_sample_agg_2hop(
                rowptr,
                col,
                features,
                roots,
                fanout1,
                fanout2,
                replay_seed=seed + split_offset * 1_000_000 + start,
            )
            root_ids = roots.long()
            logits = model(features[root_ids], mean)
            target = labels[root_ids]
            correct += int((logits.argmax(dim=-1) == target).sum())
            total += int(target.numel())
        results[split_name] = accuracy(correct, total)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["pyg", "fsa"])
    parser.add_argument(
        "--dataset", required=True, choices=["reddit", "ogbn-arxiv", "ogbn-products"]
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--fanout", nargs=2, type=int, default=[15, 10])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/parity"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset_root = args.data_root / args.dataset
    data, split, num_classes = load_dataset(args.dataset, dataset_root)
    features = data.x.float().cuda().contiguous()
    labels = data.y.cuda()
    fanout1, fanout2 = args.fanout

    torch.manual_seed(args.seed)
    model = ParityHead(features.size(1), args.hidden, num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    records = []
    training_wall = 0.0

    if args.variant == "pyg":
        loader = NeighborLoader(
            data,
            num_neighbors=[fanout1, fanout2],
            input_nodes=split["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
    else:
        rowptr, col = load_csr(dataset_root / "csr_adj.npz")
        train_nodes = split["train"].cuda()

    for epoch in range(args.epochs):
        model.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if args.variant == "pyg":
            for step, batch in enumerate(loader):
                if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                    break
                batch = batch.cuda()
                logits = model(batch.x[: batch.batch_size], pyg_nested_mean(batch))
                target = batch.y[: batch.batch_size]
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        else:
            generator = torch.Generator(device="cuda").manual_seed(args.seed + epoch)
            order = torch.randperm(train_nodes.numel(), generator=generator, device="cuda")
            batches = math.ceil(train_nodes.numel() / args.batch_size)
            for step in range(batches):
                if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                    break
                roots = train_nodes[
                    order[step * args.batch_size : (step + 1) * args.batch_size]
                ].to(torch.int32)
                mean = fused_sample_agg_2hop(
                    rowptr,
                    col,
                    features,
                    roots,
                    fanout1,
                    fanout2,
                    replay_seed=args.seed + epoch * 1_000_000 + step,
                )
                root_ids = roots.long()
                logits = model(features[root_ids], mean)
                loss = F.cross_entropy(logits, labels[root_ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        training_wall += time.perf_counter() - started

        torch.manual_seed(args.seed)
        if args.variant == "pyg":
            metrics = evaluate_pyg(
                model,
                data,
                split,
                [fanout1, fanout2],
                args.eval_batch_size,
            )
        else:
            metrics = evaluate_fsa(
                model,
                features,
                labels,
                split,
                rowptr,
                col,
                fanout1,
                fanout2,
                args.eval_batch_size,
                args.seed,
            )
        record = {
            "dataset": args.dataset,
            "variant": args.variant,
            "seed": args.seed,
            "epoch": epoch + 1,
            "training_wall_s": training_wall,
            "train_acc": metrics["train"],
            "valid_acc": metrics["valid"],
            "test_acc": metrics["test"],
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{args.dataset}_{args.variant}_seed{args.seed}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

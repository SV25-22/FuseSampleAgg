from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dgl
import dgl.function as dgl_fn
import numpy as np
import torch
from ogb.linkproppred import Evaluator, LinkPropPredDataset
from torch import nn
from torch.nn import functional as F

from fuseop import fused_sample_agg_2hop


def as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def type_names(values) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def global_ids(ids, types, metadata: dict) -> np.ndarray:
    result = as_numpy(ids).astype(np.int64, copy=True)
    names = type_names(types)
    if names is None:
        return result
    for name, offset in metadata["entity_type_offsets"].items():
        result[names == name] += int(offset)
    return result


def global_negative_ids(ids, types, metadata: dict) -> np.ndarray:
    result = as_numpy(ids).astype(np.int64, copy=True)
    names = type_names(types)
    if names is None:
        return result
    for name, offset in metadata["entity_type_offsets"].items():
        result[names == name, :] += int(offset)
    return result


def load_csr(path: Path):
    with np.load(path) as data:
        rowptr = torch.from_numpy(data["rowptr"].astype(np.int32, copy=False))
        col = torch.from_numpy(data["col"].astype(np.int32, copy=False))
    return rowptr.contiguous(), col.contiguous()


def build_dgl_graph(rowptr: torch.Tensor, col: torch.Tensor):
    degrees = (rowptr[1:] - rowptr[:-1]).to(torch.int64)
    rows = torch.arange(rowptr.numel() - 1, dtype=torch.int32).repeat_interleave(degrees)
    return dgl.graph(
        (col.cuda(), rows.cuda()),
        num_nodes=rowptr.numel() - 1,
        idtype=torch.int32,
        device="cuda",
    )


def dgl_nested_mean(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    roots: torch.Tensor,
    fanout1: int,
    fanout2: int,
) -> torch.Tensor:
    closest_graph = dgl.sampling.sample_neighbors(
        graph, roots, fanout1, edge_dir="in", replace=False
    )
    closest = dgl.to_block(closest_graph, roots)
    middle_nodes = closest.srcdata[dgl.NID]
    farthest_graph = dgl.sampling.sample_neighbors(
        graph, middle_nodes, fanout2, edge_dir="in", replace=False
    )
    farthest = dgl.to_block(farthest_graph, middle_nodes)
    middle = dgl.ops.copy_u_mean(farthest, features[farthest.srcdata[dgl.NID].long()])
    valid = farthest.in_degrees().gt(0).to(middle.dtype).unsqueeze(1)
    closest.srcdata["middle"] = middle
    closest.srcdata["valid"] = valid
    closest.update_all(dgl_fn.copy_u("middle", "message"), dgl_fn.sum("message", "sum"))
    closest.update_all(dgl_fn.copy_u("valid", "message"), dgl_fn.sum("message", "count"))
    return closest.dstdata["sum"] / closest.dstdata["count"].clamp_min(1)


class KGModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dimension: int):
        super().__init__()
        self.entities = nn.Embedding(num_entities, dimension)
        self.relations = nn.Embedding(num_relations, dimension)
        self.projection = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )

    def encode(
        self,
        entity_ids: torch.Tensor,
        variant: str,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        graph: dgl.DGLGraph | None,
        fanout1: int,
        fanout2: int,
        seed: int,
    ) -> torch.Tensor:
        unique, inverse = torch.unique(entity_ids.long(), return_inverse=True)
        if variant == "fsa":
            neighbor_mean = fused_sample_agg_2hop(
                rowptr,
                col,
                self.entities.weight,
                unique.to(torch.int32),
                fanout1,
                fanout2,
                replay_seed=seed,
            )
        else:
            if graph is None:
                raise RuntimeError("DGL graph is not initialized")
            dgl.seed(seed & 0x7FFFFFFF)
            neighbor_mean = dgl_nested_mean(
                graph, self.entities.weight, unique.to(torch.int32), fanout1, fanout2
            )
        encoded = self.projection(torch.cat([self.entities(unique), neighbor_mean], dim=-1))
        return encoded[inverse]

    @staticmethod
    def score(head, relation, tail):
        return (head * relation * tail).sum(dim=-1)


def negative_tails(
    tail_types,
    batch_indices: np.ndarray,
    count: int,
    metadata: dict,
    generator: torch.Generator,
) -> torch.Tensor:
    if tail_types is None:
        return torch.randint(
            0,
            int(metadata["num_entities"]),
            (batch_indices.size, count),
            device="cuda",
            generator=generator,
        )
    names = type_names(as_numpy(tail_types)[batch_indices])
    result = torch.empty((batch_indices.size, count), dtype=torch.long, device="cuda")
    for name in np.unique(names):
        mask = np.flatnonzero(names == name)
        size = int(metadata["entity_type_sizes"][name])
        offset = int(metadata["entity_type_offsets"][name])
        local = torch.randint(0, size, (mask.size, count), device="cuda", generator=generator)
        result[torch.from_numpy(mask).cuda()] = local + offset
    return result


@torch.no_grad()
def evaluate_mrr(
    model: KGModel,
    dataset_name: str,
    split: dict,
    which: str,
    metadata: dict,
    variant: str,
    rowptr: torch.Tensor,
    col: torch.Tensor,
    graph: dgl.DGLGraph | None,
    fanout1: int,
    fanout2: int,
    batch_size: int,
    max_edges: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    evaluator = Evaluator(name=dataset_name)
    values = split[which]
    head_types = values.get("head_type")
    tail_types = values.get("tail_type")
    heads = global_ids(values["head"], head_types, metadata)
    tails = global_ids(values["tail"], tail_types, metadata)
    relations = as_numpy(values["relation"]).astype(np.int64, copy=False)
    limit = len(heads) if max_edges <= 0 else min(len(heads), max_edges)

    def corruption_pass(side: str) -> dict[str, float]:
        negatives = global_negative_ids(
            values[f"{side}_neg"],
            head_types if side == "head" else tail_types,
            metadata,
        )
        sums: dict[str, float] = {}
        for start in range(0, limit, batch_size):
            stop = min(limit, start + batch_size)
            head = torch.from_numpy(heads[start:stop]).cuda()
            tail = torch.from_numpy(tails[start:stop]).cuda()
            relation = torch.from_numpy(relations[start:stop]).cuda()
            negative = torch.from_numpy(negatives[start:stop]).cuda()
            size, negative_count = negative.shape
            all_ids = torch.cat([head, tail, negative.reshape(-1)])
            encoded = model.encode(
                all_ids,
                variant,
                rowptr,
                col,
                graph,
                fanout1,
                fanout2,
                seed + start + (0 if side == "tail" else 10_000_000),
            )
            head_z = encoded[:size]
            tail_z = encoded[size : 2 * size]
            negative_z = encoded[2 * size :].view(size, negative_count, -1)
            relation_z = model.relations(relation)
            positive_score = model.score(head_z, relation_z, tail_z)
            if side == "tail":
                negative_score = (head_z[:, None, :] * relation_z[:, None, :] * negative_z).sum(
                    dim=-1
                )
            else:
                negative_score = (negative_z * relation_z[:, None, :] * tail_z[:, None, :]).sum(
                    dim=-1
                )
            batch_result = evaluator.eval(
                {
                    "y_pred_pos": positive_score.float().cpu(),
                    "y_pred_neg": negative_score.float().cpu(),
                }
            )
            for key, value in batch_result.items():
                name = key.removesuffix("_list")
                mean = float(torch.as_tensor(value).float().mean())
                sums[name] = sums.get(name, 0.0) + mean * size
        return {key: value / limit for key, value in sums.items()}

    available = [side for side in ("head", "tail") if f"{side}_neg" in values]
    passes = [corruption_pass(side) for side in available]
    keys = set().union(*(result.keys() for result in passes))
    return {key: sum(result[key] for result in passes) / len(passes) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["fsa", "dgl"])
    parser.add_argument("--dataset", required=True, choices=["ogbl-wikikg2", "ogbl-biokg"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--fanout", nargs=2, type=int, default=[15, 10])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--negative-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-max-edges", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    numpy_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator(device="cuda").manual_seed(args.seed)
    root = args.data_root / args.dataset
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    dataset = LinkPropPredDataset(name=args.dataset, root=str(root))
    split = dataset.get_edge_split()
    train = split["train"]
    head_types = train.get("head_type")
    tail_types = train.get("tail_type")
    heads = global_ids(train["head"], head_types, metadata)
    tails = global_ids(train["tail"], tail_types, metadata)
    relations = as_numpy(train["relation"]).astype(np.int64, copy=False)

    rowptr_cpu, col_cpu = load_csr(root / "csr_adj.npz")
    rowptr = rowptr_cpu.cuda()
    col = col_cpu.cuda()
    graph = build_dgl_graph(rowptr_cpu, col_cpu) if args.variant == "dgl" else None
    model = KGModel(
        int(metadata["num_entities"]), int(metadata["num_relations"]), args.hidden
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    fanout1, fanout2 = args.fanout

    def train_step(step: int) -> float:
        indices = numpy_rng.integers(0, len(heads), size=args.batch_size)
        head = torch.from_numpy(heads[indices]).cuda()
        tail = torch.from_numpy(tails[indices]).cuda()
        relation = torch.from_numpy(relations[indices]).cuda()
        negative = negative_tails(
            tail_types,
            indices,
            args.negative_samples,
            metadata,
            torch_rng,
        )
        endpoints = torch.cat([head, tail])
        encoded = model.encode(
            endpoints,
            args.variant,
            rowptr,
            col,
            graph,
            fanout1,
            fanout2,
            args.seed * 1_000_003 + step,
        )
        head_z, tail_z = encoded[: args.batch_size], encoded[args.batch_size :]
        relation_z = model.relations(relation)
        positive_score = model.score(head_z, relation_z, tail_z)
        negative_score = (
            head_z[:, None, :] * relation_z[:, None, :] * model.entities(negative)
        ).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score))
        loss = loss + F.binary_cross_entropy_with_logits(
            negative_score, torch.zeros_like(negative_score)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for step in range(args.warmup):
        train_step(step)
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    wall_started = time.perf_counter()
    for step in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        train_step(args.warmup + step)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            metrics = evaluate_mrr(
                model,
                args.dataset,
                split,
                "valid",
                metadata,
                args.variant,
                rowptr,
                col,
                graph,
                fanout1,
                fanout2,
                args.eval_batch_size,
                args.eval_max_edges,
                args.seed,
            )
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "wall_s": time.perf_counter() - wall_started,
                        "valid_mrr": metrics.get("mrr"),
                    },
                    sort_keys=True,
                )
            )

    peak_bytes = torch.cuda.max_memory_allocated()
    output = {
        "variant": args.variant,
        "dataset": args.dataset,
        "fanout": f"{fanout1} {fanout2}",
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "precision": "fp32",
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "median_step_ms": float(np.median(timings)),
        "peak_vram_mb": peak_bytes / (1024**2),
        "transient_peak_mb": max(0, peak_bytes - baseline_bytes) / (1024**2),
    }
    if args.evaluate:
        for split_name in ("valid", "test"):
            metrics = evaluate_mrr(
                model,
                args.dataset,
                split,
                split_name,
                metadata,
                args.variant,
                rowptr,
                col,
                graph,
                fanout1,
                fanout2,
                args.eval_batch_size,
                args.eval_max_edges,
                args.seed,
            )
            output[f"{split_name}_mrr"] = metrics.get("mrr")
    print(json.dumps(output, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

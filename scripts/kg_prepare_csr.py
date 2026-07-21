from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ogb.linkproppred import LinkPropPredDataset


def build_csr(
    num_nodes: int, rows: np.ndarray, neighbors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((neighbors, rows))
    rows = rows[order]
    neighbors = neighbors[order]
    counts = np.bincount(rows, minlength=num_nodes)
    rowptr = np.empty(num_nodes + 1, dtype=np.int64)
    rowptr[0] = 0
    np.cumsum(counts, out=rowptr[1:])
    if rowptr[-1] > np.iinfo(np.int32).max:
        raise OverflowError("the graph has too many adjacency entries for int32 CSR")
    return rowptr.astype(np.int32), neighbors.astype(np.int32, copy=False)


def type_layout(graph: dict) -> tuple[dict[str, int], dict[str, int]]:
    sizes = graph.get("num_nodes_dict")
    if not sizes:
        return {}, {}
    normalized = {str(name): int(size) for name, size in sizes.items()}
    offsets = {}
    current = 0
    for name in sorted(normalized):
        offsets[name] = current
        current += normalized[name]
    return offsets, normalized


def global_ids(ids: np.ndarray, types: np.ndarray | None, offsets: dict[str, int]) -> np.ndarray:
    result = ids.astype(np.int64, copy=True)
    if types is None:
        return result
    type_names = np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in types]
    )
    for name, offset in offsets.items():
        result[type_names == name] += offset
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ogbl-wikikg2", "ogbl-biokg"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = args.root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LinkPropPredDataset(name=args.dataset, root=str(output_dir))
    graph = dataset[0]
    split = dataset.get_edge_split()
    train = split["train"]
    heads_local = np.asarray(train["head"], dtype=np.int64)
    tails_local = np.asarray(train["tail"], dtype=np.int64)
    relations = np.asarray(train["relation"], dtype=np.int64)
    offsets, type_sizes = type_layout(graph)
    head_types = np.asarray(train["head_type"]) if "head_type" in train else None
    tail_types = np.asarray(train["tail_type"]) if "tail_type" in train else None
    heads = global_ids(heads_local, head_types, offsets)
    tails = global_ids(tails_local, tail_types, offsets)
    if type_sizes:
        num_nodes = sum(type_sizes.values())
    else:
        num_nodes = int(graph.get("num_nodes", max(heads.max(), tails.max()) + 1))

    rows = heads
    neighbors = tails
    if args.undirected:
        rows = np.concatenate([heads, tails])
        neighbors = np.concatenate([tails, heads])
    rowptr, col = build_csr(num_nodes, rows, neighbors)
    np.savez_compressed(output_dir / "csr_adj.npz", rowptr=rowptr, col=col)

    metadata = {
        "dataset": args.dataset,
        "num_entities": num_nodes,
        "num_relations": int(relations.max()) + 1,
        "entity_type_offsets": offsets,
        "entity_type_sizes": type_sizes,
        "undirected": args.undirected,
        "train_triples": int(heads.size),
        "adjacency_entries": int(col.size),
    }
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'csr_adj.npz'}")
    print(f"wrote {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

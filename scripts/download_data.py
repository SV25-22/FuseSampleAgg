from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def edge_index_to_incoming_csr(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    sources = edge_index[0].to(torch.int64).cpu().numpy()
    targets = edge_index[1].to(torch.int64).cpu().numpy()
    order = np.lexsort((sources, targets))
    rows = targets[order]
    neighbors = sources[order]
    counts = np.bincount(rows, minlength=num_nodes)
    rowptr = np.empty(num_nodes + 1, dtype=np.int64)
    rowptr[0] = 0
    np.cumsum(counts, out=rowptr[1:])
    return rowptr.astype(np.int32), neighbors.astype(np.int32, copy=False)


def save_csr(edge_index: torch.Tensor, num_nodes: int, output_dir: Path) -> None:
    rowptr, col = edge_index_to_incoming_csr(edge_index, num_nodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "csr_adj.npz"
    np.savez_compressed(output, rowptr=rowptr, col=col)
    print(f"wrote {output} ({num_nodes:,} nodes, {col.size:,} edges)")


def prepare_reddit(root: Path) -> None:
    from torch_geometric.datasets import Reddit

    output_dir = root / "reddit"
    data = Reddit(root=str(output_dir))[0]
    save_csr(data.edge_index, data.num_nodes, output_dir)


def prepare_ogbn(name: str, root: Path) -> None:
    from ogb.nodeproppred import PygNodePropPredDataset

    output_dir = root / name
    data = PygNodePropPredDataset(name=name, root=str(output_dir))[0]
    save_csr(data.edge_index, data.num_nodes, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["reddit", "ogbn-arxiv", "ogbn-products"],
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()

    for name in args.datasets:
        if name == "reddit":
            prepare_reddit(args.root)
        else:
            prepare_ogbn(name, args.root)


if __name__ == "__main__":
    main()

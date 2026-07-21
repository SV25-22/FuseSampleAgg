# FuseSampleAgg

FuseSampleAgg is the reference implementation for **“FuseSampleAgg: One-Pass
Neighborhood Estimation for Budgeted Knowledge-Graph Refresh and Validation”**
(KSEM 2026), by Aleksandar Stanković, Haoran Du, and Xinming Wang.

[Read the paper](paper/FuseSampleAgg_KSEM_2026.pdf)

The project provides a PyTorch CUDA operator that samples uniform neighbor
sets without replacement and emits their mean directly. It avoids materialized
sampled blocks and intermediate feature gathers while retaining the same
nested-mean estimator for fixed sampled neighbor IDs.

The published FP32 experiments report 2.24×–3.48× lower end-to-end step latency
than the best tested DGL mode on the node workloads, lower peak memory, and
matching task quality within seed variability on node classification and OGB
knowledge-graph completion.

## Features

- Deterministic 1-hop and 2-hop sampling from int32 CSR graphs
- Uniform reservoir sampling without replacement
- Direct FP32 mean and nested-mean aggregation
- Exact saved-index replay for gradients when features or embeddings train
- PyTorch current-stream execution and non-default-stream coverage
- Matched DGL, PyG parity, WikiKG2, and BioKG experiment workflows

## Installation

The paper environment targets Linux, Python 3.11, CUDA 12.1, and an NVIDIA GPU.
A matching CUDA toolkit must be available to compile the extension.

```bash
conda env create -f environment.yml
conda activate fusesampleagg
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

## Python API

```python
import torch
from fuseop import fused_sample_agg, fused_sample_agg_2hop

rowptr = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device="cuda")
col = torch.tensor([1, 2, 0, 1], dtype=torch.int32, device="cuda")
features = torch.randn(3, 128, device="cuda")
roots = torch.tensor([0, 2], dtype=torch.int32, device="cuda")

one_hop = fused_sample_agg(rowptr, col, features, roots, fanout=2)
two_hop = fused_sample_agg_2hop(rowptr, col, features, roots, 2, 2)
```

CSR row `v` must contain the neighbor IDs to sample for node `v`. Inputs must
be CUDA tensors on one device; graph indices and roots use `torch.int32`, and
features use `torch.float32`. The wrapper normalizes contiguous layout and
automatically saves the sampled IDs needed by backward. For explicit replay or
analysis, use `fused_sample_agg_with_samples` or
`fused_sample_agg_2hop_with_samples`.

## Repository layout

- `fuseop/`: Python API, C++ bindings, and CUDA kernels
- `tests/`: semantic, gradient, edge-case, width, and stream tests
- `train/`: matched node, parity, and KG experiment implementations
- `scripts/`: dataset preparation, experiment grids, and table generation
- `bench/`: isolated 2-hop CUDA microbenchmark
- `docs/reproduce.md`: complete paper reproduction protocol
- `paper/`: published paper

Generated datasets, raw results, and derived artifacts are intentionally
ignored. This prevents stale outputs from being mistaken for the paper's
authoritative measurements.

## Reproduction

See [docs/reproduce.md](docs/reproduce.md) for the exact data preparation,
benchmark, parity, KG-completion, and table-generation commands.

## Citation

```bibtex
@inproceedings{stankovic2026fusesampleagg,
  title     = {FuseSampleAgg: One-Pass Neighborhood Estimation for Budgeted
               Knowledge-Graph Refresh and Validation},
  author    = {Stankovi\'{c}, Aleksandar and Du, Haoran and Wang, Xinming},
  booktitle = {Knowledge Science, Engineering and Management (KSEM)},
  year      = {2026}
}
```

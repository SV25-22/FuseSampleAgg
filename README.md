# FuseSampleAgg

FuseSampleAgg is a PyTorch CUDA operator for fused neighborhood sampling and
mean aggregation without materializing sampled subgraphs.

**Paper:** [“FuseSampleAgg: One-Pass Neighborhood Estimation for Budgeted
Knowledge-Graph Refresh and Validation”](paper/FuseSampleAgg_KSEM_2026.pdf),
presented at the 19th International Conference on Knowledge Science,
Engineering and Management (KSEM 2026), by Aleksandar Stanković, Haoran Du,
and Xinming Wang. [Official publication and
DOI](https://doi.org/10.1007/978-981-92-2852-2_8)

## Results summary

In the paper's FP32 experiments, FuseSampleAgg reduced end-to-end step latency
by 2.24×–3.48× and transient GPU memory by up to 160× over the best tested DGL
mode on the node workloads. On the knowledge-graph completion workloads,
step-time speedups were 1.44×–1.86× on BioKG and 1.15×–1.16× on WikiKG2, with
ranking quality matching the DGL baseline within seed variability.

## Features

- Deterministic 1-hop and 2-hop sampling from `int32` CSR graphs
- Uniform reservoir sampling without replacement
- Direct FP32 mean and nested-mean aggregation
- Exact saved-index replay for gradients when features or embeddings train
- PyTorch current-stream execution and non-default-stream coverage
- Matched DGL, PyG parity, WikiKG2, and BioKG experiment workflows

## Requirements and installation

Reproducing the paper environment requires Linux, Python 3.11, an NVIDIA GPU,
and a CUDA 12.1 toolkit capable of compiling a C++17 PyTorch extension. The
pinned environment installs PyTorch 2.4.0, DGL 2.4.0, PyG 2.6.1, OGB 1.3.6,
and the remaining dependencies.

```bash
conda env create -f environment.yml
conda activate fusesampleagg
python -m pip install -e . --no-build-isolation
```

## Minimal API example

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

```text
fuseop/    Python API, C++ bindings, and CUDA kernels
tests/     Semantic, gradient, edge-case, width, and stream tests
train/     Matched node, parity, and knowledge-graph experiments
scripts/   Dataset preparation, experiment grids, and table generation
bench/     Isolated 2-hop CUDA microbenchmark
docs/      Complete paper reproduction protocol
paper/     KSEM 2026 paper
```

Generated datasets, raw results, and derived artifacts are intentionally
ignored so stale outputs are not mistaken for the paper's authoritative
measurements.

## Reproducing the experiments

Follow the [complete reproduction protocol](docs/reproduce.md) for the exact
environment, data preparation, node benchmarks, accuracy-parity runs,
knowledge-graph completion experiments, and derived tables. The paper reports
FP32 measurements from one NVIDIA A800-SXM4-40GB GPU; use an otherwise idle GPU
and retain the generated raw result files.

## Tests

After building the extension, run the CUDA correctness and regression suite and the static checks:

```bash
python -m pytest -q
python -m ruff check .
```

The test suite requires a CUDA-capable NVIDIA GPU and covers 1-hop and 2-hop
semantics, gradients, edge cases, arbitrary feature widths, reproducible
sampling, and non-default streams.

## Limitations

FuseSampleAgg targets the FP32, 1-hop and 2-hop GraphSAGE-mean primitive; the
current experiments do not establish FP16/BF16 behavior or universal
performance across graph frameworks and hardware. The knowledge-graph encoder
is intentionally lightweight and does not implement relation-aware,
attention-based, or deeper-hop message passing. Task metrics remain stochastic
because of neighbor and negative sampling and floating-point reduction order.

## Citation

GitHub's **Cite this repository** menu reads [`CITATION.cff`](CITATION.cff). The
equivalent BibTeX citation is:

```bibtex
@inproceedings{stankovic2026fusesampleagg,
  title     = {FuseSampleAgg: One-Pass Neighborhood Estimation for Budgeted
               Knowledge-Graph Refresh and Validation},
  author    = {Stankovi\'{c}, Aleksandar and Du, Haoran and Wang, Xinming},
  booktitle = {Knowledge Science, Engineering and Management},
  pages     = {100--110},
  year      = {2026},
  doi       = {10.1007/978-981-92-2852-2_8}
}
```

## License

The software is released under the [MIT License](LICENSE). Third-party datasets
and dependencies remain subject to their own licenses.

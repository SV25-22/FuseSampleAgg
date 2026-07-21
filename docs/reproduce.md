# Reproducing the paper experiments

The paper reports FP32 results from one NVIDIA A800-SXM4-40GB GPU. Run every
variant on an otherwise idle GPU and keep the generated raw files unchanged.

## Environment and extension

```bash
conda env create -f environment.yml
conda activate fusesampleagg
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

The pinned environment uses Python 3.11, PyTorch 2.4.0 with CUDA 12.1, and
DGL 2.4.0 with the matching PyTorch/CUDA wheel.

## Node-classification data

```bash
python scripts/download_data.py \
  --datasets reddit ogbn-arxiv ogbn-products
```

CSR row `v` stores the incoming neighbors of `v`, matching PyG's default
neighbor-sampling direction. The DGL graph is constructed with the equivalent
edge orientation.

## Node performance table

```bash
python scripts/bench_grid.py \
  --datasets reddit ogbn-arxiv ogbn-products \
  --fanouts "15 10" "25 10" \
  --batch-sizes 512 1024 \
  --steps 200 \
  --warmup 20 \
  --repeats 5 \
  --out results/node.csv
```

Each timed step includes batch retrieval, neighborhood sampling, aggregation,
forward, backward, and the optimizer update. The DGL result chosen for a
configuration is the fastest median among GPU graph, CPU workers, and CPU UVA.
Persistent allocations present before measurement are excluded from the
reported transient peak.

## Accuracy parity

Run both variants with the same seed. Repeat with seeds 42 through 46 when
reporting mean and standard deviation.

```bash
python -m train.accuracy_parity \
  --variant pyg --dataset reddit --seed 42

python -m train.accuracy_parity \
  --variant fsa --dataset reddit --seed 42
```

Replace `reddit` with `ogbn-products` for the second paper dataset. Output is
written to `results/parity/`.

## Knowledge-graph data

```bash
python scripts/kg_prepare_csr.py --dataset ogbl-wikikg2
python scripts/kg_prepare_csr.py --dataset ogbl-biokg
```

BioKG uses type-local entity IDs. The preparation script assigns a stable
global offset to each entity type and records the mapping in `meta.json`.

## Knowledge-graph performance and MRR

```bash
python scripts/kg_grid.py --mode performance
python scripts/kg_grid.py --mode quality
```

Performance mode runs 200 measured steps after 20 warmup steps for both KG
datasets, fanouts, batches, variants, and five seeds. Quality mode runs the
WikiKG2 6000-step experiments and performs final valid/test evaluation.

## Derived tables

```bash
python scripts/make_paper_artifacts.py \
  --results results \
  --out artifacts/tables
```

This produces CSV and LaTeX summaries without modifying the raw results.

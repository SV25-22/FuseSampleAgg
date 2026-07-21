from __future__ import annotations

import pytest
import torch

from fuseop import fused_sample_agg, fused_sample_agg_with_samples

from .reference import assert_unique_valid_samples, mean_from_samples, toy_graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("feature_dim", [1, 8, 31, 32, 33, 128, 129])
@pytest.mark.parametrize("fanout", [0, 1, 2, 5])
def test_forward_matches_saved_samples(feature_dim, fanout):
    rowptr, col, x, frontier = toy_graph(feature_dim)
    out, samples, takes = fused_sample_agg_with_samples(
        rowptr, col, x, frontier, fanout, replay_seed=17
    )
    torch.testing.assert_close(out, mean_from_samples(x, samples))
    torch.testing.assert_close(takes, samples.ge(0).sum(dim=1).to(torch.int32))
    if fanout:
        assert_unique_valid_samples(samples)


def test_seed_and_frontier_order_are_reproducible():
    rowptr, col, x, frontier = toy_graph(16)
    first = fused_sample_agg_with_samples(rowptr, col, x, frontier, 2, 99)
    second = fused_sample_agg_with_samples(rowptr, col, x, frontier, 2, 99)
    for lhs, rhs in zip(first, second, strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


def test_backward_automatically_captures_samples():
    rowptr, col, x, frontier = toy_graph(13)
    x.requires_grad_(True)
    out = fused_sample_agg(rowptr, col, x, frontier, fanout=3, replay_seed=42, save_indices=False)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    _, samples, _ = fused_sample_agg_with_samples(
        rowptr, col, x.detach(), frontier, fanout=3, replay_seed=42
    )
    reference = torch.zeros_like(x)
    for batch_index, row in enumerate(samples):
        valid = row[row >= 0].long()
        if valid.numel():
            reference.index_add_(
                0,
                valid,
                (grad_out[batch_index] / valid.numel()).expand(valid.numel(), -1),
            )
    torch.testing.assert_close(x.grad, reference)


def test_non_default_stream():
    rowptr, col, x, frontier = toy_graph(17)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = fused_sample_agg(rowptr, col, x, frontier, 3, replay_seed=5)
        marker = out.square().sum()
    stream.synchronize()
    assert torch.isfinite(marker)

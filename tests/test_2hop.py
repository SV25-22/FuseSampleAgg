from __future__ import annotations

import pytest
import torch

from fuseop import fused_sample_agg_2hop, fused_sample_agg_2hop_with_samples

from .reference import (
    assert_unique_valid_samples,
    nested_mean_from_samples,
    toy_graph,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("feature_dim", [1, 8, 16, 64, 127, 128, 129, 256])
def test_forward_matches_saved_samples_for_arbitrary_width(feature_dim):
    rowptr, col, x, frontier = toy_graph(feature_dim)
    out, samples1, samples2 = fused_sample_agg_2hop_with_samples(
        rowptr, col, x, frontier, 3, 2, replay_seed=1234
    )
    torch.testing.assert_close(out, nested_mean_from_samples(x, samples2))
    assert_unique_valid_samples(samples1)
    assert_unique_valid_samples(samples2)


@pytest.mark.parametrize("fanout1,fanout2", [(0, 0), (0, 2), (2, 0)])
def test_zero_fanout_returns_zero(fanout1, fanout2):
    rowptr, col, x, frontier = toy_graph(16)
    out = fused_sample_agg_2hop(rowptr, col, x, frontier, fanout1, fanout2, replay_seed=1)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_backward_automatically_replays_samples():
    rowptr, col, x, frontier = toy_graph(19)
    x.requires_grad_(True)
    out = fused_sample_agg_2hop(
        rowptr,
        col,
        x,
        frontier,
        3,
        2,
        replay_seed=7,
        save_indices=False,
    )
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    _, _, samples2 = fused_sample_agg_2hop_with_samples(
        rowptr, col, x.detach(), frontier, 3, 2, replay_seed=7
    )
    reference = torch.zeros_like(x)
    for root in range(samples2.size(0)):
        nonempty = [row[row >= 0].long() for row in samples2[root] if row.ge(0).any()]
        for valid in nonempty:
            contribution = grad_out[root] / (len(nonempty) * valid.numel())
            reference.index_add_(0, valid, contribution.expand(valid.numel(), -1))
    torch.testing.assert_close(x.grad, reference)


def test_non_default_stream():
    rowptr, col, x, frontier = toy_graph(129)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = fused_sample_agg_2hop(rowptr, col, x, frontier, 3, 2, 11)
        marker = out.square().sum()
    stream.synchronize()
    assert torch.isfinite(marker)

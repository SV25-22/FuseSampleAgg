"""PyTorch interface for FuseSampleAgg."""

from __future__ import annotations

import torch

try:
    from . import _C as fused_ext
except ImportError as exc:
    raise ImportError(
        "Build FuseSampleAgg with `python -m pip install -e . --no-build-isolation`."
    ) from exc


def _check_python_inputs(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    frontier: torch.Tensor,
) -> None:
    tensors = (rowptr, col, x, frontier)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("rowptr, col, x, and frontier must be CUDA tensors")
    if rowptr.dtype != torch.int32 or col.dtype != torch.int32:
        raise TypeError("rowptr and col must have dtype torch.int32")
    if frontier.dtype != torch.int32:
        raise TypeError("frontier must have dtype torch.int32")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")


class _FusedSampleAggFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        x: torch.Tensor,
        frontier: torch.Tensor,
        fanout: int,
        replay_seed: int,
        save_indices: bool,
    ) -> torch.Tensor:
        _check_python_inputs(rowptr, col, x, frontier)
        capture = bool(save_indices) or x.requires_grad
        rowptr = rowptr.contiguous()
        col = col.contiguous()
        x = x.contiguous()
        frontier = frontier.contiguous()

        if capture:
            out, samples, takes = fused_ext.fused_sample_agg_forward_with_samples(
                rowptr, col, x, frontier, int(fanout), int(replay_seed)
            )
            ctx.save_for_backward(samples, takes)
            ctx.num_nodes = int(x.size(0))
        else:
            out = fused_ext.fused_sample_agg_forward(
                rowptr, col, x, frontier, int(fanout), int(replay_seed)
            )
        ctx.capture = capture
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_x = None
        if ctx.capture and ctx.needs_input_grad[2]:
            samples, takes = ctx.saved_tensors
            grad_x = fused_ext.fused_sample_agg_backward(
                samples, takes, grad_out.contiguous(), ctx.num_nodes
            )
        return None, None, grad_x, None, None, None, None


def fused_sample_agg(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    frontier: torch.Tensor,
    fanout: int,
    replay_seed: int = 1234,
    save_indices: bool = False,
) -> torch.Tensor:
    """Return the sampled 1-hop neighbor mean for each frontier node."""
    return _FusedSampleAggFn.apply(rowptr, col, x, frontier, fanout, replay_seed, save_indices)


def fused_sample_agg_with_samples(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    frontier: torch.Tensor,
    fanout: int,
    replay_seed: int = 1234,
):
    """Return 1-hop means, sampled neighbor IDs, and valid sample counts."""
    _check_python_inputs(rowptr, col, x, frontier)
    return fused_ext.fused_sample_agg_forward_with_samples(
        rowptr.contiguous(),
        col.contiguous(),
        x.contiguous(),
        frontier.contiguous(),
        int(fanout),
        int(replay_seed),
    )


class _FusedSampleAgg2HopFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        x: torch.Tensor,
        frontier: torch.Tensor,
        fanout1: int,
        fanout2: int,
        replay_seed: int,
        save_indices: bool,
    ) -> torch.Tensor:
        _check_python_inputs(rowptr, col, x, frontier)
        capture = bool(save_indices) or x.requires_grad
        rowptr = rowptr.contiguous()
        col = col.contiguous()
        x = x.contiguous()
        frontier = frontier.contiguous()

        if capture:
            out, samples1, samples2 = fused_ext.fused_sample_agg_2hop_forward_with_samples(
                rowptr,
                col,
                x,
                frontier,
                int(fanout1),
                int(fanout2),
                int(replay_seed),
            )
            ctx.save_for_backward(samples1, samples2)
            ctx.num_nodes = int(x.size(0))
        else:
            out = fused_ext.fused_sample_agg_2hop_forward(
                rowptr,
                col,
                x,
                frontier,
                int(fanout1),
                int(fanout2),
                int(replay_seed),
            )
        ctx.capture = capture
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_x = None
        if ctx.capture and ctx.needs_input_grad[2]:
            samples1, samples2 = ctx.saved_tensors
            grad_x = fused_ext.fused_sample_agg_2hop_backward(
                grad_out.contiguous(), samples1, samples2, ctx.num_nodes
            )
        return None, None, grad_x, None, None, None, None, None


def fused_sample_agg_2hop(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    frontier: torch.Tensor,
    fanout1: int,
    fanout2: int,
    replay_seed: int = 1234,
    save_indices: bool = False,
) -> torch.Tensor:
    """Return the nested sampled 2-hop neighbor mean for each frontier node."""
    return _FusedSampleAgg2HopFn.apply(
        rowptr,
        col,
        x,
        frontier,
        fanout1,
        fanout2,
        replay_seed,
        save_indices,
    )


def fused_sample_agg_2hop_with_samples(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    frontier: torch.Tensor,
    fanout1: int,
    fanout2: int,
    replay_seed: int = 1234,
):
    """Return nested means and sampled first- and second-hop neighbor IDs."""
    _check_python_inputs(rowptr, col, x, frontier)
    return fused_ext.fused_sample_agg_2hop_forward_with_samples(
        rowptr.contiguous(),
        col.contiguous(),
        x.contiguous(),
        frontier.contiguous(),
        int(fanout1),
        int(fanout2),
        int(replay_seed),
    )


__all__ = [
    "fused_sample_agg",
    "fused_sample_agg_with_samples",
    "fused_sample_agg_2hop",
    "fused_sample_agg_2hop_with_samples",
]

from __future__ import annotations

import torch


def toy_graph(feature_dim: int, device: str = "cuda"):
    rowptr = torch.tensor([0, 3, 5, 6, 10, 10, 11, 14], dtype=torch.int32, device=device)
    col = torch.tensor(
        [1, 2, 3, 0, 4, 3, 1, 4, 5, 6, 6, 4, 5, 0],
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(123)
    x = torch.randn(7, feature_dim, generator=generator, device=device)
    frontier = torch.tensor([0, 3, 4, 6], dtype=torch.int32, device=device)
    return rowptr, col, x, frontier


def mean_from_samples(x: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    output = x.new_zeros((samples.size(0), x.size(1)))
    for row in range(samples.size(0)):
        valid = samples[row][samples[row] >= 0].long()
        if valid.numel():
            output[row] = x[valid].mean(dim=0)
    return output


def nested_mean_from_samples(x: torch.Tensor, samples2: torch.Tensor) -> torch.Tensor:
    output = x.new_zeros((samples2.size(0), x.size(1)))
    for root in range(samples2.size(0)):
        middle_means = []
        for middle in range(samples2.size(1)):
            valid = samples2[root, middle]
            valid = valid[valid >= 0].long()
            if valid.numel():
                middle_means.append(x[valid].mean(dim=0))
        if middle_means:
            output[root] = torch.stack(middle_means).mean(dim=0)
    return output


def assert_unique_valid_samples(samples: torch.Tensor) -> None:
    for row in samples.reshape(-1, samples.size(-1)):
        valid = row[row >= 0]
        assert valid.unique().numel() == valid.numel()

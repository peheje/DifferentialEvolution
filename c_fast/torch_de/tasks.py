"""Task generation matching C de_seq.c target functions.

Tasks:
  pair1           - 2D recurrence: x[t] = sin(2.5*x[t-1]) + 0.6*x[t-2]
  pair5-huber0.5  - 5D mostly-linear recurrence
  pair5-hard      - 5D multi-sine + interaction terms
"""

import torch


def recurrence_next(xt, xt1):
    """2D recurrence: x[t] = sin(2.5*x[t-1]) + 0.6*x[t-2]"""
    return torch.sin(2.5 * xt) + 0.6 * xt1


def recurrence_nd(xs):
    """N-dimensional mostly-linear recurrence."""
    y = torch.sin(2.5 * xs[..., 0])
    coeff = 0.6
    for i in range(1, xs.shape[-1]):
        y = y + coeff * xs[..., i]
        coeff = coeff * 0.6
    return y


def recurrence_5d_hard(xs):
    """Hard 5D regression: multi-sine + interaction term.

    y = sin(2.5*x0) + 0.7*sin(1.7*x1) + 0.35*x2*x3
      + 0.25*sin(x4 + x0) + 0.2*x2
    """
    return (torch.sin(2.5 * xs[..., 0])
            + 0.7 * torch.sin(1.7 * xs[..., 1])
            + 0.35 * xs[..., 2] * xs[..., 3]
            + 0.25 * torch.sin(xs[..., 4] + xs[..., 0])
            + 0.2 * xs[..., 2])


def generate_data(n_samples, n_dims, target_fn, lo=-2.5, hi=2.5,
                  seed=0xD1FF3A9E2B1C4D77, device="cpu", dtype=torch.float32):
    """Generate (inputs, targets) for a given target function.

    Returns tensors of shape [n_samples, n_dims] and [n_samples].
    Uses deterministic torch RNG seeded per call.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    inputs = (lo + (hi - lo) * torch.rand(n_samples, n_dims,
              generator=gen, dtype=dtype, device="cpu"))
    targets = target_fn(inputs)
    return inputs.to(device), targets.to(device)


# Task registry: name -> (n_dims, target_fn, architecture)
TASK_SPECS = {
    "pair1": {
        "n_dims": 2,
        "target_fn": lambda xs: recurrence_next(xs[..., 0], xs[..., 1]),
        "hidden_layers": 1,
        "hidden": 12,
        "outputs": 1,
    },
    "pair5-huber0.5": {
        "n_dims": 5,
        "target_fn": recurrence_nd,
        "hidden_layers": 1,
        "hidden": 16,
        "outputs": 1,
    },
    "pair5-hard": {
        "n_dims": 5,
        "target_fn": recurrence_5d_hard,
        "hidden_layers": 1,
        "hidden": 16,
        "outputs": 1,
    },
}

"""Guardrails and validation tests for the PyTorch DE harness.

Run with: python -m torch_de.validate
"""

import math
import sys

import torch

from . import mlp
from . import tasks as task_mod
from .bench import mse_loss
from .de import de_solve


def _check(condition, msg):
    if not condition:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  PASS: {msg}")


def guardrail_a_shape():
    """Shape tests: forward pass returns correct shapes, no gradients."""
    print("Guardrail A: shape tests")

    model = mlp.MLP(2, 1, 12, 1)
    pop = torch.randn(32, model.total_weights, requires_grad=False)
    inputs = torch.randn(16, 2, requires_grad=False)

    preds = model.population_forward(pop, inputs)
    _check(preds.shape == (32, 16, 1),
           f"pop forward shape {preds.shape} == (32, 16, 1)")
    _check(not preds.requires_grad, "no gradients on predictions")

    loss = mse_loss(preds, torch.randn(16))
    _check(loss.shape == (32,), f"loss shape {loss.shape} == (32,)")
    _check(not loss.requires_grad, "no gradients on loss")

    # Single forward
    single_preds = model.forward(pop[0], inputs)
    _check(single_preds.shape == (16, 1),
           f"single forward shape {single_preds.shape} == (16, 1)")

    print()


def guardrail_b_param_counts():
    """Parameter count tests matching C known counts."""
    print("Guardrail B: parameter counts")

    expected = {
        (2, 1, 12, 1): 49,
        (2, 2, 12, 1): 205,
        (5, 1, 16, 1): 113,
        (2, 1, 12, 2): 62,
    }

    for (I, L, H, O), expected_n in expected.items():
        m = mlp.MLP(I, L, H, O)
        _check(m.total_weights == expected_n,
               f"{I}->{'->'.join([str(H)] * L)}->{O}: "
               f"{m.total_weights} == {expected_n}")

    print()


def guardrail_c_trivial_sanity():
    """Trivial target sanity: zero weights, single bias, output shapes."""
    print("Guardrail C: trivial target sanity")

    model = mlp.MLP(2, 1, 4, 1)
    flat = torch.zeros(model.total_weights)
    inputs = torch.ones(8, 2)

    with torch.no_grad():
        out = model.forward(flat, inputs)

    # With zero weights and zero bias (negated 0 = 0), output should be 0
    _check(torch.allclose(out, torch.zeros_like(out), atol=1e-7),
           f"zero weights -> near-zero output (max |out| = {out.abs().max():.2e})")

    # Non-zero bias test: sigmoid(1.0) ≈ 0.7311
    # In genann format: first weight is -bias, so set flat[0] = -1.0 (bias=1.0)
    flat2 = torch.zeros(model.total_weights)
    flat2[0] = -1.0  # bias = 1.0 for first hidden neuron
    out2 = model.forward(flat2, inputs)
    _check(torch.allclose(out2, torch.zeros_like(out2), atol=1e-7),
           f"single bias, zero output weights -> near-zero (max |out| = {out2.abs().max():.2e})")

    print()


def guardrail_d_de_smoke():
    """DE smoke test: small run on pair1, verify loss improves."""
    print("Guardrail D: DE smoke test")

    task_spec = task_mod.TASK_SPECS["pair1"]
    n_dims = task_spec["n_dims"]
    target_fn = task_spec["target_fn"]

    model = mlp.MLP(n_dims, 1, 8, 1)

    train_in, train_tgt = task_mod.generate_data(
        128, n_dims, target_fn, device="cpu", dtype=torch.float32,
        seed=42)

    # Save/restore torch RNG state (not used by DE, but be safe)
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1234)

    def objective(pop):
        preds = model.population_forward(pop, train_in)
        return mse_loss(preds, train_tgt)

    pop_init = torch.rand(32, model.total_weights) * 10 - 5  # [-5, 5]
    init_losses = objective(pop_init)
    init_best = float(init_losses.min())

    best_out = torch.zeros(model.total_weights)
    loss = de_solve(
        model.total_weights, 32, 32, 20,
        -5.0, 5.0,
        objective, 42, best_out,
        mutation_strategy=0,
        device="cpu",
        dtype=torch.float32,
    )

    _check(loss < init_best,
           f"DE improved loss: {loss:.6g} < init_best {init_best:.6g}")
    _check(torch.isfinite(best_out).all(), "best solution is finite")
    _check((best_out >= -5.0).all() and (best_out <= 5.0).all(),
           "best solution within bounds")

    torch.random.set_rng_state(rng_state)
    print()


def guardrail_e_strategy_smoke():
    """Strategy smoke tests: verify rand/1, best/1, basepbest50 all run."""
    print("Guardrail E: strategy smoke tests")

    task_spec = task_mod.TASK_SPECS["pair1"]
    model = mlp.MLP(2, 1, 8, 1)

    train_in, train_tgt = task_mod.generate_data(
        128, 2, task_spec["target_fn"], device="cpu", dtype=torch.float32,
        seed=42)

    def objective(pop):
        preds = model.population_forward(pop, train_in)
        return mse_loss(preds, train_tgt)

    for strategy, label in [(0, "rand/1"), (1, "best/1"), (5, "basepbest50")]:
        pbest = 50 if strategy == 5 else 10
        loss = de_solve(
            model.total_weights, 32, 32, 30,
            -5.0, 5.0,
            objective, 42,
            mutation_strategy=strategy,
            pbest_percent=pbest,
            device="cpu",
            dtype=torch.float32,
        )
        _check(math.isfinite(loss),
               f"{label}: loss is finite ({loss:.6g})")
        _check(loss > 0, f"{label}: loss > 0 ({loss:.6g})")

    print()


def guardrail_f_structural():
    """Structural determinism: forward pass consistency, pop vs individual.

    This is NOT a C-numerical comparison.  Exact C comparison requires
    an exported fixture from the C side (identical weights and inputs
    in genann layout).  This test verifies that the PyTorch forward
    pass is internally consistent: deterministic, matmul decomposition
    matches MLP.forward, and population_forward matches per-individual.
    """
    print("Guardrail F: structural determinism tests")

    # Determinism: same weights + same inputs = same output
    model = mlp.MLP(2, 1, 12, 1)
    torch.manual_seed(12345)
    flat = torch.randn(model.total_weights) * 2
    inputs = torch.randn(32, 2)

    out1 = model.forward(flat, inputs)
    out2 = model.forward(flat, inputs)

    _check(torch.allclose(out1, out2, atol=1e-15),
           f"deterministic forward pass (max diff = {(out1 - out2).abs().max():.2e})")

    # Unpack_flat + manual matmul matches MLP.forward
    Ws, Bs = mlp.unpack_flat(flat, 2, 1, 12, 1)
    x = inputs
    for k in range(len(Ws) - 1):
        x = torch.sigmoid(x @ Ws[k].T + Bs[k])
    manual_out = x @ Ws[-1].T + Bs[-1]

    _check(torch.allclose(out1, manual_out, atol=1e-6),
           f"manual matmul matches MLP.forward (max diff = {(out1 - manual_out).abs().max():.2e})")

    # Population forward vs individual forward
    pop = torch.randn(8, model.total_weights)
    pop_preds = model.population_forward(pop, inputs)
    for i in range(8):
        indiv = model.forward(pop[i], inputs)
        diff = (pop_preds[i] - indiv).abs().max()
        _check(diff < 1e-5,
               f"pop[{i}] matches individual forward (max diff = {diff:.2e})")

    print()
    print("NOTE: Structural determinism verified. Not a C-numerical comparison.")
    print("      C fixture export required for exact cross-harness validation.")


def run_all():
    """Run all guardrails. Exits with code 1 on first failure."""
    print("=" * 60)
    print("PyTorch DE Guardrails")
    print("=" * 60)
    print()

    guardrail_a_shape()
    guardrail_b_param_counts()
    guardrail_c_trivial_sanity()
    guardrail_d_de_smoke()
    guardrail_e_strategy_smoke()
    guardrail_f_structural()

    print("=" * 60)
    print("All guardrails passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_all()

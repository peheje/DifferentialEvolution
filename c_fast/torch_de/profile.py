"""Performance profiling for torch_de DE solver.

Usage: python -m torch_de.profile
"""

import time

import numpy as np
import torch

from . import mlp
from . import tasks as task_mod
from .bench import mse_loss, BENCHMARKS
from .de import de_solve, _pick_all_indices, _lpsr_population_size


# === instrumented DE solver (timing breakdown) ===

def de_solve_instrumented(
    params, initial_popsize, min_popsize, generations,
    bounds_min, bounds_max, objective_fn, seed,
    best_out=None, mutation_strategy=0, pbest_percent=10,
    device="cuda", dtype=torch.float32,
):
    """Copy of de_solve with per-section timing."""
    if params <= 0 or initial_popsize < 4 or min_popsize < 4:
        return float("inf"), {}
    if min_popsize > initial_popsize:
        min_popsize = initial_popsize

    use_lpsr = min_popsize < initial_popsize
    P = initial_popsize
    D = params

    rng = np.random.Generator(np.random.PCG64(seed))
    bound_lo = float(bounds_min)
    bound_hi = float(bounds_max)

    pop_np = rng.uniform(bound_lo, bound_hi, size=(P, D))
    pop = torch.from_numpy(pop_np).to(dtype=dtype, device=device)

    Fi = torch.from_numpy(
        rng.uniform(0.1, 0.9, size=P)).to(dtype=dtype, device=device)
    CRi = torch.from_numpy(
        rng.uniform(0.0, 1.0, size=P)).to(dtype=dtype, device=device)

    scores = objective_fn(pop)

    # Cumulative timers
    t_rank = 0.0
    t_jde = 0.0
    t_index = 0.0
    t_mutate = 0.0
    t_objective = 0.0
    t_select = 0.0
    t_lpsr = 0.0

    for g in range(generations):
        active = P

        # --- ranking ---
        t0 = time.perf_counter()
        sorted_cpu = torch.argsort(scores).cpu().numpy()
        best_idx = int(sorted_cpu[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_rank += time.perf_counter() - t0

        pbest_count = 1
        if mutation_strategy == 5:
            pbest_count = max(1, (active * pbest_percent + 99) // 100)

        # --- jDE update ---
        t0 = time.perf_counter()
        Fi_np = Fi.cpu().numpy().copy()
        CRi_np = CRi.cpu().numpy().copy()

        re_F = rng.random(active) < 0.1
        re_CR = rng.random(active) < 0.1
        if re_F.any():
            Fi_np[re_F] = rng.uniform(0.1, 0.9, size=re_F.sum())
        if re_CR.any():
            CRi_np[re_CR] = rng.uniform(0.0, 1.0, size=re_CR.sum())

        Fi_try = torch.from_numpy(Fi_np).to(dtype=dtype, device=device)
        CRi_try = torch.from_numpy(CRi_np).to(dtype=dtype, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_jde += time.perf_counter() - t0

        # --- index sampling (vectorized) ---
        t0 = time.perf_counter()
        if mutation_strategy == 0:
            idx_all = _pick_all_indices(rng, active, 3)
            idx_r1 = idx_all[:, 0]; idx_r2 = idx_all[:, 1]; idx_r3 = idx_all[:, 2]
            pbest_sel = None

        elif mutation_strategy == 1:
            best_excl = np.full(active, best_idx, dtype=np.int32)
            idx_all = _pick_all_indices(rng, active, 2, best_excl)
            idx_r1 = idx_all[:, 0]; idx_r2 = idx_all[:, 1]
            idx_r3 = None; pbest_sel = None

        elif mutation_strategy == 5:
            pbest_sel = sorted_cpu[rng.integers(0, pbest_count, size=active)]
            if pbest_count > 1:
                self_mask = pbest_sel == np.arange(active)
                for _ in range(8):
                    if not self_mask.any():
                        break
                    pbest_sel[self_mask] = sorted_cpu[
                        rng.integers(0, pbest_count, size=self_mask.sum())]
                    self_mask = pbest_sel == np.arange(active)

            idx_all = _pick_all_indices(rng, active, 2, pbest_sel)
            idx_r1 = idx_all[:, 0]; idx_r2 = idx_all[:, 1]
            idx_r3 = None
        t_index += time.perf_counter() - t0

        # --- mutation / crossover / clamp ---
        t0 = time.perf_counter()
        if mutation_strategy == 0:
            r1_t = torch.from_numpy(idx_r1).to(device=device)
            r2_t = torch.from_numpy(idx_r2).to(device=device)
            r3_t = torch.from_numpy(idx_r3).to(device=device)
            base = pop[r1_t]
            diff = pop[r2_t] - pop[r3_t]
        elif mutation_strategy == 1:
            r1_t = torch.from_numpy(idx_r1).to(device=device)
            r2_t = torch.from_numpy(idx_r2).to(device=device)
            base = pop[best_idx].unsqueeze(0).expand(active, -1)
            diff = pop[r1_t] - pop[r2_t]
        elif mutation_strategy == 5:
            pb_t = torch.from_numpy(pbest_sel).to(device=device)
            r1_t = torch.from_numpy(idx_r1).to(device=device)
            r2_t = torch.from_numpy(idx_r2).to(device=device)
            base = pop[pb_t]
            diff = pop[r1_t] - pop[r2_t]

        mutant = base + Fi_try.unsqueeze(1) * diff

        cross_rand = torch.from_numpy(
            rng.random((active, D)).astype(np.float32)
        ).to(dtype=dtype, device=device)
        cross_mask = cross_rand < CRi_try.unsqueeze(1)

        jrand_vals = torch.from_numpy(
            rng.integers(0, D, size=active)).to(device=device)
        cross_mask[torch.arange(active, device=device), jrand_vals] = True

        trials = torch.where(cross_mask, mutant, pop)
        trials = torch.clamp(trials, bound_lo, bound_hi)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_mutate += time.perf_counter() - t0

        # --- objective evaluation ---
        t0 = time.perf_counter()
        trial_scores = objective_fn(trials)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_objective += time.perf_counter() - t0

        # --- selection ---
        t0 = time.perf_counter()
        improved = trial_scores < scores
        improved_idx = improved.nonzero(as_tuple=False).squeeze(-1)
        if improved_idx.numel() > 0:
            pop[improved_idx] = trials[improved_idx]
            scores[improved_idx] = trial_scores[improved_idx]
            Fi[improved_idx] = Fi_try[improved_idx]
            CRi[improved_idx] = CRi_try[improved_idx]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_select += time.perf_counter() - t0

        # --- LPSR ---
        t0 = time.perf_counter()
        if use_lpsr:
            next_p = _lpsr_population_size(initial_popsize, min_popsize,
                                          generations, g)
            if next_p < P:
                sorted_idx = torch.argsort(scores)
                keep = sorted_idx[:next_p]
                pop = pop[keep]
                scores = scores[keep]
                Fi = Fi[keep]
                CRi = CRi[keep]
                P = next_p
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_lpsr += time.perf_counter() - t0

    best_idx = int(torch.argmin(scores).item())
    best_loss = float(scores[best_idx])
    if best_out is not None:
        best_out.copy_(pop[best_idx])

    total = t_rank + t_jde + t_index + t_mutate + t_objective + t_select + t_lpsr
    breakdown = {
        "rank": t_rank, "jde": t_jde, "index": t_index,
        "mutate": t_mutate, "objective": t_objective,
        "select": t_select, "lpsr": t_lpsr, "total": total,
    }
    return best_loss, breakdown


# === objective with internal breakdown ===

def instrumented_objective(pop, model, inputs, targets, loss_type="mse",
                           huber_delta=None):
    """Objective function with internal timing for unpack/forward/loss."""
    t0 = time.perf_counter()
    Ws, Bs = mlp.unpack_flat(pop, model.inputs, model.hidden_layers,
                             model.hidden, model.outputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_unpack = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Manually replicate population_forward for timing
    P = pop.shape[0]
    x = inputs.T.unsqueeze(0).expand(P, -1, -1)  # [P, I, S]
    for k in range(len(Ws)):
        x = torch.bmm(Ws[k], x)
        x = x + Bs[k].unsqueeze(-1)
        x = x.permute(0, 2, 1)
        if k < len(Ws) - 1:
            x = torch.sigmoid(x)
            x = x.permute(0, 2, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_forward = time.perf_counter() - t0

    t0 = time.perf_counter()
    if loss_type == "huber":
        t = targets.view(1, -1, 1)
        e = x - t
        ae = torch.abs(e)
        mse_part = 0.5 * e * e
        linear_part = huber_delta * (ae - 0.5 * huber_delta)
        loss = torch.where(ae <= huber_delta, mse_part, linear_part).mean(dim=(1, 2))
    else:
        t = targets.view(1, -1, 1)
        e = x - t
        loss = (e * e).mean(dim=(1, 2))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_loss = time.perf_counter() - t0

    return loss, {"unpack": t_unpack, "forward": t_forward, "loss": t_loss}


# === microbenchmarks ===

def micro_eval(model_config, pop_size, samples, device="cuda",
               dtype=torch.float32, n_warmup=5, n_iters=30):
    """Microbenchmark population_forward + loss at a given scale."""
    I, L, H, O = model_config
    model = mlp.MLP(I, L, H, O)
    D = model.total_weights

    # Dummy data
    torch.manual_seed(42)
    pop = torch.randn(pop_size, D, dtype=dtype, device=device) * 2.5
    inputs = torch.randn(samples, I, dtype=dtype, device=device) * 2.5
    targets = torch.randn(samples, dtype=dtype, device=device)

    for _ in range(n_warmup):
        model.population_forward(pop, inputs)
        mse_loss(preds := model.population_forward(pop, inputs), targets)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        preds = model.population_forward(pop, inputs)
        loss = mse_loss(preds, targets)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    best = times[0]
    return {"best_ms": best, "median_ms": median, "pop_size": pop_size,
            "samples": samples, "model": f"{I}->{'->'.join([str(H)]*L)}->{O}",
            "total_weights": D}


def run_microbenchmarks(device="cuda"):
    """Run eval-only microbenchmarks at various scales."""
    configs = [
        # (label, model_config, pop_size, samples)
        ("current (pair1)", (2, 1, 12, 1), 384, 512),
        ("current (pair5)", (5, 1, 16, 1), 768, 512),
        ("wider", (5, 1, 64, 1), 768, 512),
        ("deeper", (5, 2, 64, 1), 768, 512),
        ("larger pop", (5, 1, 16, 1), 4096, 512),
        ("larger samples", (5, 1, 16, 1), 768, 4096),
        ("large everything", (5, 2, 64, 1), 4096, 4096),
    ]

    print("=" * 80)
    print("Eval-only microbenchmarks (population_forward + MSE loss)")
    print(f"Device: {device}, dtype: float32")
    print(f"{'Config':<22} {'Model':<20} {'Weights':<8} {'P':<8} {'S':<8} "
          f"{'Best(ms)':<10} {'Median(ms)':<10} {'MEvals/s'}")

    for label, cfg, P, S in configs:
        r = micro_eval(cfg, P, S, device=device)
        mevals_s = (P * S) / (r["median_ms"] / 1000) / 1e6
        print(f"{label:<22} {r['model']:<20} {r['total_weights']:<8} "
              f"{P:<8} {S:<8} {r['best_ms']:<10.3f} {r['median_ms']:<10.3f} "
              f"{mevals_s:>8.2f}")
    print()


# === torch.compile test ===

def benchmark_compiled(device="cuda"):
    """Test torch.compile on the objective function."""
    print("=" * 80)
    print("torch.compile comparison (eval-only, pair5-hard scale)")
    print(f"Device: {device}")

    model = mlp.MLP(5, 1, 16, 1)
    D = model.total_weights
    P, S = 768, 512

    torch.manual_seed(42)
    pop = torch.randn(P, D, dtype=torch.float32, device=device) * 2.5
    inputs = torch.randn(S, 5, dtype=torch.float32, device=device) * 2.5
    targets = torch.randn(S, dtype=torch.float32, device=device)

    def compute(p):
        preds = model.population_forward(p, inputs)
        t = targets.view(1, -1, 1)
        e = preds - t
        return (e * e).mean(dim=(1, 2))

    # Eager: warmup + timed
    for _ in range(5):
        compute(pop)
    if device == "cuda":
        torch.cuda.synchronize()
    eager_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        compute(pop)
        if device == "cuda":
            torch.cuda.synchronize()
        eager_times.append((time.perf_counter() - t0) * 1000)
    eager_times.sort()

    # Compile
    print("  compiling... (this may take a while)")
    t0 = time.perf_counter()
    compute_compiled = torch.compile(compute, mode="max-autotune")
    # Warmup (compile on first call)
    compute_compiled(pop)
    if device == "cuda":
        torch.cuda.synchronize()
    compile_time = (time.perf_counter() - t0) * 1000

    compiled_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        compute_compiled(pop)
        if device == "cuda":
            torch.cuda.synchronize()
        compiled_times.append((time.perf_counter() - t0) * 1000)
    compiled_times.sort()

    print(f"  Compile time: {compile_time:.0f}ms")
    print(f"  {'':<20} {'Best(ms)':<12} {'Median(ms)':<12}")
    print(f"  {'eager':<20} {eager_times[0]:<12.3f} {eager_times[len(eager_times)//2]:<12.3f}")
    print(f"  {'compiled':<20} {compiled_times[0]:<12.3f} {compiled_times[len(compiled_times)//2]:<12.3f}")
    print()


# === scatter-free MLP ===

class MLPSlice(mlp.MLP):
    """MLP variant that slices flat vectors directly, no scatter."""

    def population_forward(self, pop, inputs):
        P, D = pop.shape
        I = self.inputs
        H = self.hidden
        O = self.outputs
        L = self.hidden_layers
        S = inputs.shape[0]

        idx = 0
        x = inputs.T.unsqueeze(0).expand(P, -1, -1)  # [P, I, S]

        for h in range(L):
            prev = I if h == 0 else H
            # Skip biases, reshape weights
            b = -pop[:, idx:idx + H]                    # [P, H]
            idx += H
            W = pop[:, idx:idx + H * prev].view(P, H, prev)  # [P, H, prev]
            idx += H * prev

            x = torch.bmm(W, x) + b.unsqueeze(-1)       # [P, H, S]
            x = x.permute(0, 2, 1)                        # [P, S, H]
            x = self.activation(x)
            x = x.permute(0, 2, 1)                        # [P, H, S]

        # Output layer
        out_prev = H if L > 0 else I
        b = -pop[:, idx:idx + O]                          # [P, O]
        idx += O
        W = pop[:, idx:idx + O * out_prev].view(P, O, out_prev)  # [P, O, out_prev]

        x = torch.bmm(W, x) + b.unsqueeze(-1)             # [P, O, S]
        x = x.permute(0, 2, 1)                             # [P, S, O]
        return x


def benchmark_slice_vs_scatter(device="cuda"):
    """Compare scatter (unpack_flat) vs direct slice forward pass."""
    print("=" * 80)
    print("Scatter (unpack_flat) vs slice (direct view) comparison")
    print(f"Device: {device}")

    configs = [
        ("pair1 scale", (2, 1, 12, 1), 384, 512),
        ("pair5 scale", (5, 1, 16, 1), 768, 512),
        ("wider", (5, 1, 64, 1), 768, 512),
        ("deeper", (5, 2, 64, 1), 768, 512),
    ]

    for label, (I, L, H, O), P, S in configs:
        model_scatter = mlp.MLP(I, L, H, O)
        model_slice = MLPSlice(I, L, H, O)
        D = model_scatter.total_weights

        torch.manual_seed(42)
        pop = torch.randn(P, D, dtype=torch.float32, device=device) * 2.5
        inputs = torch.randn(S, I, dtype=torch.float32, device=device) * 2.5
        targets = torch.randn(S, dtype=torch.float32, device=device)

        # Verify correctness
        with torch.no_grad():
            out_scatter = model_scatter.population_forward(pop, inputs)
            out_slice = model_slice.population_forward(pop, inputs)
        max_diff = (out_scatter - out_slice).abs().max().item()

        # Benchmark scatter
        for _ in range(5):
            model_scatter.population_forward(pop, inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        scatter_times = []
        for _ in range(30):
            t0 = time.perf_counter()
            model_scatter.population_forward(pop, inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            scatter_times.append((time.perf_counter() - t0) * 1000)
        scatter_times.sort()

        # Benchmark slice
        for _ in range(5):
            model_slice.population_forward(pop, inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        slice_times = []
        for _ in range(30):
            t0 = time.perf_counter()
            model_slice.population_forward(pop, inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            slice_times.append((time.perf_counter() - t0) * 1000)
        slice_times.sort()

        scat_med = scatter_times[len(scatter_times) // 2]
        sli_med = slice_times[len(slice_times) // 2]
        speedup = scat_med / sli_med if sli_med > 0 else float("inf")

        print(f"  {label:<15}: scatter={scat_med:.3f}ms slice={sli_med:.3f}ms "
              f"speedup={speedup:.2f}x diff={max_diff:.2e}")


# === instrumentation run ===

def run_instrumented(device="cuda"):
    """Run instrumented DE for 200 generations and report breakdown."""
    print()
    print("=" * 80)
    print("Instrumented DE: pair5-hard-basepbest50, 200 gens")
    print(f"Device: {device}")

    task_spec = task_mod.TASK_SPECS["pair5-hard"]
    n_dims = task_spec["n_dims"]
    target_fn = task_spec["target_fn"]
    model = mlp.MLP(5, 1, 16, 1)
    D = model.total_weights

    train_in, train_tgt = task_mod.generate_data(
        512, n_dims, target_fn, device=device, dtype=torch.float32)

    # Count objective calls for detail breakdown
    obj_calls = [0]
    obj_unpack_acc = [0.0]
    obj_forward_acc = [0.0]
    obj_loss_acc = [0.0]

    def objective(pop):
        obj_calls[0] += 1
        loss, bd = instrumented_objective(pop, model, train_in, train_tgt,
                                          loss_type="mse")
        obj_unpack_acc[0] += bd["unpack"]
        obj_forward_acc[0] += bd["forward"]
        obj_loss_acc[0] += bd["loss"]
        return loss

    best_out = torch.zeros(D, dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    loss, breakdown = de_solve_instrumented(
        D, 768, 384, 200,
        -5.0, 5.0,
        objective, 42, best_out,
        mutation_strategy=5, pbest_percent=50,
        device=device, dtype=torch.float32,
    )
    wall = (time.perf_counter() - t0) * 1000

    total_bd = breakdown["total"]
    print()
    print(f"  Wall time: {wall:.0f}ms  Generations: 200")
    print(f"  Objective calls: {obj_calls[0]}")
    print()
    print(f"  {'Section':<20} {'Time(ms)':<12} {'%':<8}")
    print(f"  {'-'*38}")
    for name in ["rank", "jde", "index", "mutate", "objective",
                 "select", "lpsr"]:
        t = breakdown[name] * 1000
        pct = t / wall * 100
        print(f"  {name:<20} {t:<12.1f} {pct:<8.1f}")
    print(f"  {'-'*38}")
    print(f"  {'total (tracked)':<20} {total_bd*1000:<12.1f} {total_bd*1000/wall*100:<8.1f}")
    print(f"  {'untracked':<20} {wall - total_bd*1000:<12.1f} {(wall - total_bd*1000)/wall*100:<8.1f}")
    print()

    if obj_calls[0] > 0:
        n = obj_calls[0]
        print(f"  Objective breakdown (avg per call, {n} calls):")
        print(f"  {'  unpack':<20} {obj_unpack_acc[0]*1000/n:<10.3f}ms")
        print(f"  {'  forward':<20} {obj_forward_acc[0]*1000/n:<10.3f}ms")
        print(f"  {'  loss':<20} {obj_loss_acc[0]*1000/n:<10.3f}ms")
    print()


# === main ===

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile torch_de performance")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("PyTorch DE Performance Profile")
    print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    print()

    # C: eval-only microbenchmarks
    run_microbenchmarks(device)

    # D: torch.compile test
    if device == "cuda":
        benchmark_compiled(device)

    # E: scatter vs slice
    benchmark_slice_vs_scatter(device)

    # A: instrumented DE run
    run_instrumented(device)

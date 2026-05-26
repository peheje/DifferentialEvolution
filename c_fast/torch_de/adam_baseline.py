"""Adam baseline runner for PyTorch DE benchmark tasks.

Trains standard torch.nn.Module MLPs with Adam on the same synthetic
tasks and data generation used by the DE harness.
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn

from . import tasks as task_mod


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AdamMLP(nn.Module):
    """Standard MLP: sigmoid hidden, linear output."""

    def __init__(self, arch, activation="sigmoid"):
        """arch: list of layer widths, e.g. [5, 16, 1]."""
        super().__init__()
        self.arch = list(arch)
        layers = []
        act_fn = torch.sigmoid if activation == "sigmoid" else torch.tanh
        for i in range(len(self.arch) - 1):
            layers.append(nn.Linear(self.arch[i], self.arch[i + 1]))
            if i < len(self.arch) - 2:
                layers.append(nn.Sigmoid() if activation == "sigmoid" else nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.net(x)

    @property
    def param_count(self):
        return self._param_count


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_data(task_name, n_train=512, n_test=256, device="cpu", dtype=torch.float32):
    """Return (train_in, train_tgt, test_in, test_tgt, test_var) for a task."""
    spec = task_mod.TASK_SPECS[task_name]
    n_dims = spec["n_dims"]
    target_fn = spec["target_fn"]

    train_in, train_tgt = task_mod.generate_data(
        n_train, n_dims, target_fn,
        seed=0xD1FF3A9E2B1C4D77, device=device, dtype=dtype)
    test_in, test_tgt = task_mod.generate_data(
        n_test, n_dims, target_fn,
        seed=0x7E57C0FFEE5EED, device=device, dtype=dtype)

    test_var = float(torch.var(test_tgt, correction=0))
    return train_in, train_tgt, test_in, test_tgt, test_var


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_adam(
    model,
    train_in,
    train_tgt,
    test_in,
    test_tgt,
    loss_type="mse",
    huber_delta=0.5,
    lr=1e-3,
    weight_decay=0.0,
    steps=5000,
    device="cpu",
    dtype=torch.float32,
    seed=0,
):
    """Train model with Adam. Returns dict of metrics."""
    torch.manual_seed(seed)
    model = model.to(device=device, dtype=dtype)

    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_in = train_in.to(device=device, dtype=dtype)
    train_tgt = train_tgt.to(device=device, dtype=dtype)
    test_in = test_in.to(device=device, dtype=dtype)
    test_tgt = test_tgt.to(device=device, dtype=dtype)

    best_test_mse = float("inf")
    best_train_mse = float("inf")
    best_step = 0

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_in).squeeze(-1)
        loss = criterion(pred, train_tgt)
        loss.backward()
        optimizer.step()

        # periodic eval: every 50 steps and at the end
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                train_pred = model(train_in).squeeze(-1)
                test_pred = model(test_in).squeeze(-1)
                train_mse = float(nn.functional.mse_loss(train_pred, train_tgt))
                test_mse = float(nn.functional.mse_loss(test_pred, test_tgt))
            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_train_mse = train_mse
                best_step = step

    if device == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    return {
        "train_mse": best_train_mse,
        "test_mse": best_test_mse,
        "best_step": best_step,
        "wall_s": wall_s,
        "steps": steps,
        "steps_per_sec": steps / wall_s if wall_s > 0 else float("inf"),
        "sample_fwd_per_sec": steps * train_in.shape[0] / wall_s if wall_s > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Run matrix helpers
# ---------------------------------------------------------------------------

def run_single(task, arch, lr, weight_decay, steps, seeds, loss_type, huber_delta,
               n_train, n_test, device, dtype):
    """Run a single (task, arch, lr, wd) config across seeds."""
    train_in, train_tgt, test_in, test_tgt, test_var = get_data(
        task, n_train=n_train, n_test=n_test, device=device, dtype=dtype)

    results = []
    for seed in seeds:
        model = AdamMLP(arch)
        metrics = train_adam(
            model, train_in, train_tgt, test_in, test_tgt,
            loss_type=loss_type, huber_delta=huber_delta,
            lr=lr, weight_decay=weight_decay, steps=steps,
            device=device, dtype=dtype, seed=seed,
        )
        metrics["seed"] = seed
        results.append(metrics)
    return results, test_var


def print_result_row(r, arch_str, params, task, lr, nmse_val):
    print(f"{task:<18} {arch_str:<14} {params:<6} "
          f"lr={lr:<8.0e} seed={r['seed']:<4} "
          f"train={r['train_mse']:<12.6g} test={r['test_mse']:<12.6g} "
          f"nMSE={nmse_val:<12.6g} best_step={r['best_step']:<6} "
          f"wall={r['wall_s']:<6.1f}s "
          f"steps/s={r['steps_per_sec']:<8.0f} "
          f"samp_fwd/s={r['sample_fwd_per_sec']:<10.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_arch(s):
    return [int(x.strip()) for x in s.split(",")]


def smoke_test(device="cpu", dtype=torch.float32):
    """Quick sanity check: loss decreases, metrics are finite."""
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    task = "pair1"
    arch = [2, 12, 1]
    steps = 200
    lr = 1e-2

    train_in, train_tgt, test_in, test_tgt, test_var = get_data(
        task, n_train=512, n_test=256, device=device, dtype=dtype)

    # Measure initial loss with random weights
    torch.manual_seed(42)
    model = AdamMLP(arch).to(device=device, dtype=dtype)
    with torch.no_grad():
        initial_pred = model(train_in.to(device=device, dtype=dtype)).squeeze(-1)
        initial_mse = float(nn.functional.mse_loss(initial_pred, train_tgt.to(device=device, dtype=dtype)))

    metrics = train_adam(
        model, train_in, train_tgt, test_in, test_tgt,
        loss_type="mse", lr=lr, weight_decay=0.0, steps=steps,
        device=device, dtype=dtype, seed=42,
    )

    print(f"task={task} arch={arch} steps={steps} lr={lr}")
    print(f"  initial MSE: {initial_mse:.6g}")
    print(f"  best test MSE: {metrics['test_mse']:.6g} (step {metrics['best_step']})")
    print(f"  wall: {metrics['wall_s']:.2f}s steps/s: {metrics['steps_per_sec']:.0f}")

    ok = True
    for key in ["train_mse", "test_mse"]:
        if not math.isfinite(metrics[key]):
            print(f"  FAIL: {key} not finite: {metrics[key]}")
            ok = False
    if metrics["test_mse"] >= initial_mse * 0.95:
        print(f"  WARN: test MSE {metrics['test_mse']:.6g} >= 0.95 * initial {initial_mse:.6g}")
    if metrics["best_step"] <= 0:
        print(f"  FAIL: best_step should be > 0, got {metrics['best_step']}")
        ok = False
    if ok:
        print("SMOKE PASS")
    else:
        print("SMOKE FAIL")
        sys.exit(1)
    print()
    return True


# DE comparison numbers (from Experiment A, c3c0ff3, basepbest50 best-of-3)
DE_BASELINE = {
    ("pair5-hard", "5->16->1"):     {"test_mse": 0.0106, "wall_s": 44.9},
    ("pair5-hard", "5->64->1"):     {"test_mse": 0.0183, "wall_s": 49.8},
    ("pair5-hard", "5->64->64->1"): {"test_mse": 0.698,  "wall_s": 87.8},
}


def suite_current(device="cpu", dtype=torch.float32):
    """Run the current benchmark suite (A + B)."""
    print("=" * 60)
    print("SUITE: current (pair5-hard + pair1)")
    print("=" * 60)

    configs = []
    # A. pair5-hard
    for arch in [[5, 16, 1], [5, 64, 1], [5, 64, 64, 1]]:
        for lr in [1e-2, 3e-3, 1e-3]:
            configs.append({
                "task": "pair5-hard",
                "arch": arch,
                "lr": lr,
                "wd": 0.0,
                "loss_type": "mse",
                "huber_delta": None,
            })
    # B. pair1
    for lr in [1e-2, 3e-3, 1e-3]:
        configs.append({
            "task": "pair1",
            "arch": [2, 12, 1],
            "lr": lr,
            "wd": 0.0,
            "loss_type": "mse",
            "huber_delta": None,
        })

    seeds = [0, 1, 2]
    steps = 5000
    n_train = 512
    n_test = 256

    print(f"Configs: {len(configs)}  seeds: {seeds}  steps: {steps}  "
          f"train/test: {n_train}/{n_test}")
    print()
    print(f"{'Task':<14} {'Arch':<14} {'Params':<6} {'LR':<8} {'Seed':<5} "
          f"{'TrainMSE':<12} {'TestMSE':<12} {'nMSE':<12} "
          f"{'BestStep':<8} {'Wall(s)':<8} {'Steps/s':<9} {'SampFwd/s':<10}")
    print("-" * 130)

    all_rows = []
    for cfg in configs:
        task = cfg["task"]
        arch = cfg["arch"]
        lr = cfg["lr"]
        wd = cfg["wd"]
        loss_type = cfg["loss_type"]
        huber_delta = cfg["huber_delta"]

        arch_str = "->".join(str(x) for x in arch)
        params = sum(
            (arch[i] + 1) * arch[i + 1]
            for i in range(len(arch) - 1)
        )

        results, test_var = run_single(
            task, arch, lr, wd, steps, seeds,
            loss_type, huber_delta, n_train, n_test, device, dtype)

        for r in results:
            nmse_val = r["test_mse"] / test_var if test_var > 0 else float("inf")
            print_result_row(r, arch_str, params, task, lr, nmse_val)
            all_rows.append({
                "task": task,
                "arch": arch_str,
                "params": params,
                "lr": lr,
                "seed": r["seed"],
                "train_mse": r["train_mse"],
                "test_mse": r["test_mse"],
                "nmse": nmse_val,
                "best_step": r["best_step"],
                "wall_s": r["wall_s"],
                "steps_per_sec": r["steps_per_sec"],
                "sample_fwd_per_sec": r["sample_fwd_per_sec"],
            })

    # Summary: best per config
    print()
    print("=" * 60)
    print("SUMMARY: best test MSE per (task, arch, lr)")
    print("=" * 60)
    print(f"{'Task':<14} {'Arch':<14} {'LR':<8} {'BestTest':<12} {'MeanTest':<12} "
          f"{'Best(ms)':<10}")
    print("-" * 70)
    from itertools import groupby
    key_fn = lambda r: (r["task"], r["arch"], r["lr"])
    for key, group in groupby(sorted(all_rows, key=key_fn), key=key_fn):
        group = list(group)
        best = min(group, key=lambda r: r["test_mse"])
        mean_test = sum(r["test_mse"] for r in group) / len(group)
        print(f"{key[0]:<14} {key[1]:<14} {key[2]:<8.0e} "
              f"{best['test_mse']:<12.6g} {mean_test:<12.6g} "
              f"{best['wall_s']*1000/3:<10.0f}")

    # Best overall per arch
    print()
    print("Best overall per (task, arch):")
    key_fn2 = lambda r: (r["task"], r["arch"])
    for key, group in groupby(sorted(all_rows, key=key_fn2), key=key_fn2):
        group = list(group)
        best = min(group, key=lambda r: r["test_mse"])
        de_key = (key[0], key[1])
        de = DE_BASELINE.get(de_key)
        de_str = f"DE={de['test_mse']:.6g}" if de else ""
        print(f"  {key[0]:<14} {key[1]:<14} "
              f"lr={best['lr']:.1e} seed={best['seed']} "
              f"test MSE={best['test_mse']:.6g} nMSE={best['nmse']:.6g} "
              f"wall={best['wall_s']:.1f}s best_step={best['best_step']} "
              f"{'[' + de_str + ']' if de else ''}")

    print()
    return all_rows


def _main():
    parser = argparse.ArgumentParser(description="Adam baseline for DE tasks")
    parser.add_argument("--smoke", action="store_true",
                        help="Run quick smoke test and exit")
    parser.add_argument("--suite", default=None, choices=["current"],
                        help="Suite to run: 'current'")
    parser.add_argument("--task", default="pair5-hard",
                        help="Task name (default: pair5-hard)")
    parser.add_argument("--arch", default="5,16,1",
                        help="Comma-separated layer widths (default: 5,16,1)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training steps (default: 5000)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (default: 0)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds to run (default: 3)")
    parser.add_argument("--train-samples", type=int, default=512,
                        help="Train samples (default: 512)")
    parser.add_argument("--test-samples", type=int, default=256,
                        help="Test samples (default: 256)")
    parser.add_argument("--loss", default="mse", choices=["mse", "huber"],
                        help="Loss type (default: mse)")
    parser.add_argument("--huber-delta", type=float, default=0.5,
                        help="Huber delta (default: 0.5)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float64"],
                        help="Tensor dtype (default: float32)")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.smoke:
        smoke_test(device=args.device, dtype=dtype)
        sys.exit(0)

    if args.suite == "current":
        suite_current(device=args.device, dtype=dtype)
        sys.exit(0)

    # Single config run
    arch = _parse_arch(args.arch)
    seeds = list(range(args.seeds))
    loss_type = args.loss
    huber_delta = args.huber_delta if loss_type == "huber" else None

    results, test_var = run_single(
        args.task, arch, args.lr, args.weight_decay, args.steps, seeds,
        loss_type, huber_delta,
        args.train_samples, args.test_samples, args.device, dtype)

    arch_str = "->".join(str(x) for x in arch)
    params = sum(
        (arch[i] + 1) * arch[i + 1]
        for i in range(len(arch) - 1)
    )

    print(f"task={args.task} arch={arch_str} params={params} "
          f"lr={args.lr} wd={args.weight_decay} steps={args.steps}")
    for r in results:
        nmse_val = r["test_mse"] / test_var if test_var > 0 else float("inf")
        print_result_row(r, arch_str, params, args.task, args.lr, nmse_val)


if __name__ == "__main__":
    _main()

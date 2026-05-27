"""Non-neural baselines for the synthetic DE/Adam regression tasks.

Compares Adam MLPs against tabular regressors and genetic-programming symbolic
regression on the same deterministic train/test fixtures.
"""

import argparse
import math
import time
import warnings

import torch

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gplearn.genetic import SymbolicRegressor

from .adam_baseline import AdamMLP, get_data, train_adam


warnings.filterwarnings("ignore", message=".*ill-conditioned matrix.*")
warnings.filterwarnings("ignore", message=".*Singular matrix.*")


def _parse_arch(value):
    return [int(part.strip()) for part in value.split(",")]


def _to_numpy(train_in, train_tgt, test_in, test_tgt):
    return (
        train_in.detach().cpu().numpy(),
        train_tgt.detach().cpu().numpy(),
        test_in.detach().cpu().numpy(),
        test_tgt.detach().cpu().numpy(),
    )


def _mse(model, x, y):
    pred = model.predict(x)
    return float(mean_squared_error(y, pred))


def run_sklearn_model(name, model, x_train, y_train, x_test, y_test, test_var):
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    wall_s = time.perf_counter() - t0
    train_mse = _mse(model, x_train, y_train)
    test_mse = _mse(model, x_test, y_test)
    return {
        "method": name,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "nmse": test_mse / test_var,
        "wall_s": wall_s,
        "extra": "",
    }


def run_kernel_ridge_grid(x_train, y_train, x_test, y_test, test_var, args):
    best = None
    t0 = time.perf_counter()
    for gamma in args.rbf_gammas:
        for alpha in args.rbf_alphas:
            model = make_pipeline(
                StandardScaler(),
                KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma),
            )
            model.fit(x_train, y_train)
            train_mse = _mse(model, x_train, y_train)
            test_mse = _mse(model, x_test, y_test)
            row = {
                "method": "KernelRidge RBF grid",
                "train_mse": train_mse,
                "test_mse": test_mse,
                "nmse": test_mse / test_var,
                "wall_s": time.perf_counter() - t0,
                "extra": f"gamma={gamma:g} alpha={alpha:g}",
            }
            if best is None or row["test_mse"] < best["test_mse"]:
                best = row
    return best


def run_adam(task, arch, train_in, train_tgt, test_in, test_tgt, test_var, args):
    model = AdamMLP(arch)
    metrics = train_adam(
        model,
        train_in,
        train_tgt,
        test_in,
        test_tgt,
        loss_type="mse",
        lr=args.adam_lr,
        weight_decay=args.adam_weight_decay,
        steps=args.adam_steps,
        device=args.device,
        dtype=torch.float32,
        seed=args.seed,
    )
    return {
        "method": f"Adam MLP {','.join(str(x) for x in arch)}",
        "train_mse": metrics["train_mse"],
        "test_mse": metrics["test_mse"],
        "nmse": metrics["test_mse"] / test_var,
        "wall_s": metrics["wall_s"],
        "extra": f"best_step={metrics['best_step']}",
    }


def symbolic_program(model):
    program = getattr(model, "_program", None)
    return str(program) if program is not None else ""


def build_models(args):
    return [
        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                max_iter=args.hgb_iter,
                learning_rate=0.05,
                l2_regularization=1e-4,
                random_state=args.seed,
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=args.trees,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=args.seed,
                n_jobs=args.jobs,
            ),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=args.trees,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=args.seed,
                n_jobs=args.jobs,
            ),
        ),
        (
            "Symbolic GP",
            SymbolicRegressor(
                population_size=args.gp_population,
                generations=args.gp_generations,
                tournament_size=20,
                stopping_criteria=1e-5,
                const_range=(-3.0, 3.0),
                init_depth=(2, 6),
                init_method="half and half",
                function_set=("add", "sub", "mul", "div", "sin", "cos"),
                metric="mse",
                parsimony_coefficient=args.gp_parsimony,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=1.0,
                verbose=0,
                random_state=args.seed,
                n_jobs=args.jobs,
            ),
        ),
    ]


def print_row(row):
    print(
        f"{row['method']:<22} train={row['train_mse']:<12.6g} "
        f"test={row['test_mse']:<12.6g} nMSE={row['nmse']:<12.6g} "
        f"wall={row['wall_s']:<8.2f}s {row['extra']}"
    )


def run_suite(args):
    arch = _parse_arch(args.arch)
    torch_device = args.device if args.device == "cuda" else "cpu"
    train_in, train_tgt, test_in, test_tgt, test_var = get_data(
        args.task,
        n_train=args.train_samples,
        n_test=args.test_samples,
        device=torch_device,
        dtype=torch.float32,
    )
    x_train, y_train, x_test, y_test = _to_numpy(
        train_in, train_tgt, test_in, test_tgt
    )

    print("=" * 88)
    print("Non-NN Baseline Experiment")
    print("=" * 88)
    print(
        f"task={args.task} arch={arch} train={args.train_samples} "
        f"test={args.test_samples} device={args.device} seed={args.seed}"
    )
    print()

    rows = []
    if "adam" in args.methods or "all" in args.methods:
        rows.append(run_adam(
            args.task, arch, train_in, train_tgt, test_in, test_tgt, test_var, args
        ))
        print_row(rows[-1])

    for name, model in build_models(args):
        key = name.lower().split()[0]
        if key not in args.methods and "all" not in args.methods:
            continue
        row = run_sklearn_model(name, model, x_train, y_train, x_test, y_test, test_var)
        if name == "Symbolic GP":
            program = symbolic_program(model)
            row["extra"] = f"program={program[:160]}"
        rows.append(row)
        print_row(row)

    if "kernelridge" in args.methods or "all" in args.methods:
        row = run_kernel_ridge_grid(x_train, y_train, x_test, y_test, test_var, args)
        rows.append(row)
        print_row(row)

    rows.sort(key=lambda r: r["test_mse"])
    print()
    print("Ranked by test MSE")
    for idx, row in enumerate(rows, start=1):
        print(f"{idx}. {row['method']}: test={row['test_mse']:.6g} nMSE={row['nmse']:.6g}")

    if any(not math.isfinite(r["test_mse"]) for r in rows):
        raise RuntimeError("non-finite test MSE in result rows")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pair5-hard", choices=["pair1", "pair5-huber0.5", "pair5-hard"])
    parser.add_argument("--arch", default="5,64,1")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all", "adam"],
        choices=["all", "adam", "histgradientboosting", "extratrees", "randomforest", "kernelridge", "symbolic"],
    )
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--adam-steps", type=int, default=5000)
    parser.add_argument("--adam-lr", type=float, default=3e-3)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--hgb-iter", type=int, default=1000)
    parser.add_argument("--rbf-gammas", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--rbf-alphas", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--gp-population", type=int, default=2000)
    parser.add_argument("--gp-generations", type=int, default=25)
    parser.add_argument("--gp-parsimony", type=float, default=0.001)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    run_suite(args)


if __name__ == "__main__":
    _main()

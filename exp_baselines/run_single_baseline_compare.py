import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append("exp_baselines")
sys.path.append("exp_baselines/bo_models")
sys.path.append("exp_warmstarting")

from bayesmark.sklearn_funcs import SklearnModelCustom
from tasks import bo_loop


BASELINES = {
    "random": "bo_random",
    "tpe": "bo_tpe",
    "optuna": "bo_optuna",
    "smac": "bo_smac",
    "hebo": "bo_hebo",
    "skopt": "bo_skopt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="digits")
    parser.add_argument("--model", default="RF")
    parser.add_argument("--metric", default="acc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=25)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["random", "tpe", "optuna", "smac"],
        choices=sorted(BASELINES),
    )
    parser.add_argument(
        "--out-dir",
        default="exp_baselines/results_single_compare/digits_RF_seed0",
    )
    args = parser.parse_args()

    smc_object = SklearnModelCustom(args.model, args.dataset, args.metric)
    config_space, order_list = smc_object.get_config_space()
    fun_to_evaluate = smc_object.obtain_evaluate(smc_object.evaluate)

    init_path = Path("init-configs") / smc_object.path_name / "Random" / f"config{args.seed}.json"
    with init_path.open() as f:
        config_init = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": args.dataset,
        "model": args.model,
        "metric": args.metric,
        "seed": args.seed,
        "n_init": args.n_init,
        "n_runs": args.n_runs,
        "init_path": str(init_path),
        "baselines": [],
    }

    for baseline in args.baselines:
        bo_type = BASELINES[baseline]
        print("=" * 100, flush=True)
        print(f"Running baseline={baseline} bo_type={bo_type}", flush=True)
        try:
            _, all_metrics_pd = bo_loop(
                bo_type,
                n_repetitions=1,
                fun_to_evaluate=fun_to_evaluate,
                config_space=config_space,
                order_list=order_list,
                n_runs=args.n_runs,
                n_init=args.n_init,
                list_init_config=[config_init],
            )
        except Exception as exc:
            print(f"FAILED baseline={baseline}: {type(exc).__name__}: {exc}", flush=True)
            manifest["baselines"].append(
                {"name": baseline, "bo_type": bo_type, "status": "failed", "error": repr(exc)}
            )
            continue

        metrics = all_metrics_pd[0].copy()
        metrics.insert(0, "eval_index", range(len(metrics)))
        metrics.insert(1, "phase", ["init" if i < args.n_init else "bo" for i in range(len(metrics))])
        metrics["cv_score"] = -metrics["obj_loss"]
        metrics["baseline"] = baseline
        csv_path = out_dir / f"{baseline}.csv"
        metrics.to_csv(csv_path, index=False)

        summary = {
            "name": baseline,
            "bo_type": bo_type,
            "status": "ok",
            "csv": str(csv_path),
            "num_evals": int(len(metrics)),
            "best_cv_score": float(metrics["cv_score"].max()),
            "best_generalization_score": float(metrics["generalization_score"].max()),
        }
        manifest["baselines"].append(summary)
        print(summary, flush=True)

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print("=" * 100, flush=True)
    print(f"Wrote {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

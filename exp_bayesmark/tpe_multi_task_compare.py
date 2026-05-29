import argparse
import json
import math
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from bayesmark.bbox_utils import get_bayesmark_func
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


TASK_MAP = {
    "breast": ("classification", "accuracy"),
    "digits": ("classification", "accuracy"),
    "wine": ("classification", "accuracy"),
    "iris": ("classification", "accuracy"),
    "diabetes": ("regression", "neg_mean_squared_error"),
}


DEFAULT_TASKS = [
    ("digits", "SVM"),
    ("digits", "MLP_SGD"),
    ("wine", "RandomForest"),
    ("wine", "SVM"),
    ("wine", "AdaBoost"),
    ("breast", "DecisionTree"),
    ("breast", "RandomForest"),
    ("iris", "MLP_SGD"),
    ("diabetes", "RandomForest"),
    ("diabetes", "SVM"),
]


def suggest_config(trial, constraints):
    cfg = {}
    for name, (value_type, transform, bounds) in constraints.items():
        lo, hi = bounds
        if value_type == "int":
            cfg[name] = trial.suggest_int(name, int(lo), int(hi))
        elif value_type == "float":
            cfg[name] = trial.suggest_float(name, float(lo), float(hi), log=(transform == "log"))
        else:
            raise ValueError(f"Unsupported value type for {name}: {value_type}")
    return cfg


def evaluate_config(cfg, data, constraints, model_name, task_type, metric, seed):
    np.random.seed(seed)
    random.seed(seed)
    fixed = dict(cfg)
    for name, value in fixed.items():
        if constraints[name][0] == "int":
            fixed[name] = int(value)

    x_train, x_test = data["train_x"], data["test_x"]
    y_train, y_test = data["train_y"], data["test_y"]
    if task_type == "regression":
        mean_ = np.mean(y_train)
        std_ = np.std(y_train)
        y_train = (y_train - mean_) / std_
        y_test = (y_test - mean_) / std_

    bbox_func = get_bayesmark_func(model_name, task_type, y_test)
    scorer = get_scorer(metric)
    model = bbox_func(**fixed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_score = float(np.mean(cross_val_score(model, x_train, y_train, scoring=scorer, cv=5)))

    model = bbox_func(**fixed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(x_train, y_train)
        generalization_score = float(scorer(model, x_test, y_test))

    if metric == "neg_mean_squared_error":
        cv_score = -cv_score
        generalization_score = -generalization_score
    return cv_score, generalization_score


def summarize_task(task_dir, dataset, model, metric):
    results = pd.read_csv(task_dir / "results.csv")
    lower = metric == "neg_mean_squared_error"
    score = results["cv_score"]
    gen = results["generalization_score"]
    best_idx = score.idxmin() if lower else score.idxmax()
    worst_idx = score.idxmax() if lower else score.idxmin()
    param_cols = [c for c in results.columns if c.startswith("param__")]
    summary = {
        "dataset": dataset,
        "model": model,
        "method": "TPE",
        "metric": "mean_squared_error" if lower else "accuracy",
        "lower_is_better": lower,
        "n_trials": int(len(results)),
        "unique_config_count": int(results[param_cols].drop_duplicates().shape[0]),
        "mean_cv": float(score.mean()),
        "median_cv": float(score.median()),
        "std_cv": float(score.std(ddof=0)),
        "best_cv": float(score.loc[best_idx]),
        "worst_cv": float(score.loc[worst_idx]),
        "mean_generalization": float(gen.mean()),
        "median_generalization": float(gen.median()),
        "best_generalization_at_best_cv": float(gen.loc[best_idx]),
        "worst_generalization_at_worst_cv": float(gen.loc[worst_idx]),
    }
    with open(task_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_task(args, dataset, model, hp_config):
    task_type, metric = TASK_MAP[dataset]
    constraints = hp_config[model]
    lower = metric == "neg_mean_squared_error"
    direction = "minimize" if lower else "maximize"
    task_id = f"{dataset}_{model}"
    task_dir = Path(args.out_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    with open(f"bayesmark/data/{dataset}.pickle", "rb") as f:
        data = pickle.load(f)

    rows = []

    def objective(trial):
        cfg = suggest_config(trial, constraints)
        cv, gen = evaluate_config(cfg, data, constraints, model, task_type, metric, args.seed)
        objective_value = cv if not lower else cv
        record = {
            "dataset": dataset,
            "model": model,
            "task_type": task_type,
            "metric": "mean_squared_error" if lower else metric,
            "trial": trial.number,
            "cv_score": cv,
            "generalization_score": gen,
        }
        for name, value in cfg.items():
            record[f"param__{name}"] = value
        rows.append(record)
        pd.DataFrame(rows).to_csv(task_dir / "results.csv", index=False)
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    summary = summarize_task(task_dir, dataset, model, metric)
    print(f"TPE_TASK_COMPLETE {task_id} {summary}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="exp_bayesmark/results_tpe_multitask")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="*", default=[f"{d}:{m}" for d, m in DEFAULT_TASKS])
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open("hp_configurations/bayesmark.json") as f:
        hp_config = json.load(f)

    task_pairs = []
    for item in args.tasks:
        dataset, model = item.split(":", 1)
        if dataset not in TASK_MAP:
            raise ValueError(f"unknown dataset: {dataset}")
        if model not in hp_config:
            raise ValueError(f"unknown model: {model}")
        task_pairs.append((dataset, model))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "task_list.json", "w") as f:
        json.dump([{"dataset": d, "model": m} for d, m in task_pairs], f, indent=2)

    summaries = []
    for dataset, model in task_pairs:
        summaries.append(run_task(args, dataset, model, hp_config))
        pd.DataFrame(summaries).to_csv(out_dir / "summary_all_tasks.csv", index=False)

    with open(out_dir / "summary_all_tasks.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("TPE_MULTI_TASK_COMPLETE", flush=True)
    print(pd.DataFrame(summaries), flush=True)


if __name__ == "__main__":
    main()

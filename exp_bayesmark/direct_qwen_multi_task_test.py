import argparse
import asyncio
import json
import math
import os
import pickle
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from aiohttp import ClientSession
from bayesmark.bbox_utils import get_bayesmark_func
from llambo.llm_client import chat_completion, get_request_timeout
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


def feature_names(constraints):
    return list(constraints.keys())


def format_ranges(constraints):
    lines = []
    for name, (value_type, transform, bounds) in constraints.items():
        lines.append(f"- {name}: [{bounds[0]}, {bounds[1]}] ({value_type}, {transform})")
    return "\n".join(lines)


def schema_line(names, constraints):
    parts = []
    for name in names:
        value_type = constraints[name][0]
        placeholder = "<int>" if value_type == "int" else "<float>"
        parts.append(f"{name}: {placeholder}")
    return "## " + ", ".join(parts) + " ##"


def build_prompt(dataset_name, model_name, task_type, metric, constraints, data):
    names = feature_names(constraints)
    n_train = data["train_x"].shape[0]
    n_features = data["train_x"].shape[1]

    if task_type == "classification":
        n_classes = len(np.unique(data["train_y"]))
        task_description = f"{dataset_name} dataset, {n_classes}-class classification"
        objective = "maximize 5-fold cross-validation accuracy"
        metric_note = "Accuracy is higher-is-better."
    else:
        task_description = f"{dataset_name} dataset, regression"
        objective = "minimize 5-fold cross-validation mean squared error"
        metric_note = "Mean squared error is lower-is-better."

    return f"""You are a careful machine learning hyperparameter optimization assistant.

Task: sklearn {task_description}.
Model: {model_name}.
Objective: {objective}.
Training set: {n_train} samples, {n_features} numerical features.

Suggest one hyperparameter configuration that you believe is most likely to perform well.

Allowed hyperparameter ranges:
{format_ranges(constraints)}

Rules:
- {metric_note}
- Stay strictly inside the allowed ranges.
- Boundary or near-boundary values are allowed when they are useful.
- Use the exact parameter names shown in the schema; do not invent alternative names.
- Do not explain your reasoning.
- Return only one configuration on one line.
- Do not output Markdown bullets, prose, code fences, or extra text.

Your response must exactly follow this schema:
{schema_line(names, constraints)}"""


def parse_config(text, names):
    parts = text.split("##")
    content = parts[1].strip() if len(parts) > 1 else text.strip()
    result = {}
    for pair in content.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            key, value = [x.strip() for x in pair.split(":", 1)]
        elif " is " in pair:
            key, value = [x.strip() for x in pair.split(" is ", 1)]
        else:
            raise ValueError(f"bad assignment: {pair}")
        if key not in names:
            raise ValueError(f"unknown key: {key}")
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value)
        if not match:
            raise ValueError(f"bad value: {value}")
        result[key] = float(match.group(0))
    missing = [name for name in names if name not in result]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    return result


def raw_in_range(cfg, constraints):
    bad = []
    for name, value in cfg.items():
        lo, hi = constraints[name][2]
        if not (float(lo) <= float(value) <= float(hi)):
            bad.append(name)
    return len(bad) == 0, bad


def clip_config(cfg, constraints):
    clipped = {}
    for name, value in cfg.items():
        value_type, _, bounds = constraints[name]
        lo, hi = float(bounds[0]), float(bounds[1])
        value = min(max(float(value), lo), hi)
        if value_type == "int":
            value = int(round(value))
        clipped[name] = value
    return clipped


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


async def ask_qwen(session, model, prompt, n, timeout):
    resp = await chat_completion(
        session=session,
        engine=model,
        messages=[
            {"role": "system", "content": "You are a careful machine learning hyperparameter tuning assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=220,
        top_p=0.95,
        n=n,
        request_timeout=timeout,
    )
    return [choice["message"]["content"] for choice in resp["choices"]]


def task_is_lower_better(metric):
    return metric == "neg_mean_squared_error"


def summarize_task(task_dir, dataset, model, metric):
    path = task_dir / "results.csv"
    results = pd.read_csv(path)
    ok = results[results["status"] == "ok"].copy()
    lower = task_is_lower_better(metric)
    summary = {
        "dataset": dataset,
        "model": model,
        "metric": "mean_squared_error" if lower else "accuracy",
        "lower_is_better": lower,
        "n_requested": int(len(results)),
        "n_ok": int(len(ok)),
        "parse_success_rate": float(len(ok) / len(results)) if len(results) else math.nan,
    }
    if not ok.empty:
        feature_cols = [c for c in ok.columns if c.startswith("param__")]
        score = ok["cv_score"]
        gen = ok["generalization_score"]
        best_idx = score.idxmin() if lower else score.idxmax()
        worst_idx = score.idxmax() if lower else score.idxmin()
        summary.update(
            {
                "raw_in_range_rate": float(ok["raw_in_range"].astype(bool).mean()),
                "unique_config_count": int(ok[feature_cols].drop_duplicates().shape[0]),
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
        )
    with open(task_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


async def run_task(session, args, dataset, model, hp_config):
    task_type, metric = TASK_MAP[dataset]
    constraints = hp_config[model]
    names = feature_names(constraints)
    task_id = f"{dataset}_{model}"
    task_dir = Path(args.out_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    with open(f"bayesmark/data/{dataset}.pickle", "rb") as f:
        data = pickle.load(f)

    prompt = build_prompt(dataset, model, task_type, metric, constraints, data)
    (task_dir / "prompt.txt").write_text(prompt + "\n")

    rows = []
    remaining = args.n
    request_idx = 0
    while remaining > 0:
        batch_n = min(args.batch_size, remaining)
        texts = await ask_qwen(session, args.model, prompt, batch_n, get_request_timeout(240))
        for batch_item_idx, text in enumerate(texts):
            record = {
                "dataset": dataset,
                "model": model,
                "task_type": task_type,
                "metric": "mean_squared_error" if metric == "neg_mean_squared_error" else metric,
                "sample_idx": len(rows),
                "request_idx": request_idx,
                "batch_item_idx": batch_item_idx,
                "raw_response": text,
            }
            try:
                raw_cfg = parse_config(text, names)
                in_range, bad_keys = raw_in_range(raw_cfg, constraints)
                cfg = clip_config(raw_cfg, constraints)
                cv, gen = evaluate_config(cfg, data, constraints, model, task_type, metric, args.seed)
                record.update(
                    {
                        "status": "ok",
                        "raw_in_range": in_range,
                        "raw_out_of_range_keys": ",".join(bad_keys),
                        "cv_score": cv,
                        "generalization_score": gen,
                    }
                )
                for name in names:
                    record[f"raw__{name}"] = raw_cfg[name]
                    record[f"param__{name}"] = cfg[name]
            except Exception as exc:
                record.update(
                    {
                        "status": "failed",
                        "raw_in_range": False,
                        "raw_out_of_range_keys": "",
                        "cv_score": math.nan,
                        "generalization_score": math.nan,
                        "error": repr(exc),
                    }
                )
            rows.append(record)
            pd.DataFrame(rows).to_csv(task_dir / "results.csv", index=False)
        request_idx += 1
        remaining -= batch_n

    summary = summarize_task(task_dir, dataset, model, metric)
    print(f"TASK_COMPLETE {task_id} {summary}")
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="exp_bayesmark/results_direct_qwen_multitask")
    parser.add_argument("--model", default=os.environ.get("LLAMBO_MODEL", "qwen3.5:9b"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="*", default=[f"{d}:{m}" for d, m in DEFAULT_TASKS])
    args = parser.parse_args()

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
    async with ClientSession(trust_env=True) as session:
        for dataset, model in task_pairs:
            summaries.append(await run_task(session, args, dataset, model, hp_config))
            pd.DataFrame(summaries).to_csv(out_dir / "summary_all_tasks.csv", index=False)

    with open(out_dir / "summary_all_tasks.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("DIRECT_QWEN_MULTI_TASK_COMPLETE")
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    asyncio.run(main())

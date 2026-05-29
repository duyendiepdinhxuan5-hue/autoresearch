import argparse
import asyncio
import json
import math
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from aiohttp import ClientSession
from bayesmark.bbox_utils import get_bayesmark_func
from llambo.llm_client import chat_completion, get_request_timeout
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


FEATURES = [
    "max_depth",
    "max_features",
    "min_impurity_decrease",
    "min_samples_leaf",
    "min_samples_split",
    "min_weight_fraction_leaf",
]


def build_prompt(constraints, data):
    ranges = []
    for name in FEATURES:
        value_type, _, bounds = constraints[name]
        ranges.append(f"- {name}: [{bounds[0]}, {bounds[1]}] ({value_type})")

    n_train = data["train_x"].shape[0]
    n_features = data["train_x"].shape[1]
    n_classes = len(np.unique(data["train_y"]))

    return f"""You are a careful machine learning hyperparameter optimization assistant.

Task: sklearn digits dataset, {n_classes}-class handwritten digit classification.
Model: RandomForestClassifier.
Objective: maximize 5-fold cross-validation accuracy.
Training set: {n_train} samples, {n_features} numerical features.

Suggest one hyperparameter configuration that you believe is most likely to achieve high accuracy.

Allowed hyperparameter ranges:
{chr(10).join(ranges)}

Rules:
- Accuracy is higher-is-better.
- Stay strictly inside the allowed ranges.
- Boundary or near-boundary values are allowed when they are useful.
- Use RandomForest knowledge for small image-like tabular classification data.
- Use the exact parameter names shown in the schema; do not invent alternative names.
- Do not explain your reasoning.
- Return only one configuration on one line.
- Do not output Markdown bullets, prose, code fences, or extra text.

Your response must exactly follow this schema:
## max_depth: <int>, max_features: <float>, min_impurity_decrease: <float>, min_samples_leaf: <float>, min_samples_split: <float>, min_weight_fraction_leaf: <float> ##"""


def parse_config(text):
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
        if key not in FEATURES:
            raise ValueError(f"unknown key: {key}")
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value)
        if not match:
            raise ValueError(f"bad value: {value}")
        result[key] = float(match.group(0))

    missing = [name for name in FEATURES if name not in result]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    return result


def clip_config(cfg, constraints):
    clipped = {}
    for name in FEATURES:
        value = float(cfg[name])
        value_type, _, bounds = constraints[name]
        lo, hi = float(bounds[0]), float(bounds[1])
        value = min(max(value, lo), hi)
        if value_type == "int":
            value = int(round(value))
        clipped[name] = value
    return clipped


def evaluate_config(cfg, data, constraints, bbox_func, seed):
    np.random.seed(seed)
    random.seed(seed)
    fixed = dict(cfg)
    for name, value in fixed.items():
        if constraints[name][0] == "int":
            fixed[name] = int(value)

    scorer = get_scorer("accuracy")
    model = bbox_func(**fixed)
    cv_score = float(np.mean(cross_val_score(model, data["train_x"], data["train_y"], scoring=scorer, cv=5)))

    model = bbox_func(**fixed)
    model.fit(data["train_x"], data["train_y"])
    gen_score = float(scorer(model, data["test_x"], data["test_y"]))
    return cv_score, gen_score


async def ask_qwen(session, model, prompt, n, timeout):
    resp = await chat_completion(
        session=session,
        engine=model,
        messages=[
            {"role": "system", "content": "You are a careful machine learning hyperparameter tuning assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=180,
        top_p=0.95,
        n=n,
        request_timeout=timeout,
    )
    return [choice["message"]["content"] for choice in resp["choices"]]


def summarize(results, out_dir):
    ok = results[results["status"] == "ok"].copy()
    summary = {
        "n_requested": int(len(results)),
        "n_ok": int(len(ok)),
        "parse_success_rate": float(len(ok) / len(results)) if len(results) else math.nan,
    }
    if not ok.empty:
        summary.update(
            {
                "mean_cv": float(ok["cv_score"].mean()),
                "median_cv": float(ok["cv_score"].median()),
                "std_cv": float(ok["cv_score"].std(ddof=0)),
                "best_cv": float(ok["cv_score"].max()),
                "worst_cv": float(ok["cv_score"].min()),
                "mean_generalization": float(ok["generalization_score"].mean()),
                "median_generalization": float(ok["generalization_score"].median()),
                "best_generalization": float(ok["generalization_score"].max()),
                "worst_generalization": float(ok["generalization_score"].min()),
                "unique_config_count": int(ok[FEATURES].drop_duplicates().shape[0]),
            }
        )
        best = ok.sort_values(["cv_score", "generalization_score"], ascending=False).iloc[0]
        summary["best_config"] = {name: best[name].item() if hasattr(best[name], "item") else best[name] for name in FEATURES}

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Direct Qwen Best Hyperparameter Test",
        "",
        "Qwen is asked to directly propose the best RandomForest hyperparameters for digits classification, without history or Bayesian optimization.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        if key == "best_config":
            continue
        lines.append(f"- {key}: {value}")
    if "best_config" in summary:
        lines.extend(["", "## Best Config", ""])
        for name, value in summary["best_config"].items():
            lines.append(f"- {name}: {value}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="exp_bayesmark/results_direct_qwen_best/digits_RF_seed0")
    parser.add_argument("--model", default=os.environ.get("LLAMBO_MODEL", "qwen3.5:9b"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("hp_configurations/bayesmark.json") as f:
        constraints = json.load(f)["RandomForest"]
    with open("bayesmark/data/digits.pickle", "rb") as f:
        data = pickle.load(f)

    prompt = build_prompt(constraints, data)
    (out_dir / "prompt.txt").write_text(prompt + "\n")

    bbox_func = get_bayesmark_func("RandomForest", "classification", data["test_y"])
    rows = []
    async with ClientSession(trust_env=True) as session:
        remaining = args.n
        request_idx = 0
        while remaining > 0:
            batch_n = min(args.batch_size, remaining)
            texts = await ask_qwen(session, args.model, prompt, batch_n, get_request_timeout(240))
            for batch_item_idx, text in enumerate(texts):
                sample_idx = len(rows)
                record = {
                    "sample_idx": sample_idx,
                    "request_idx": request_idx,
                    "batch_item_idx": batch_item_idx,
                    "raw_response": text,
                }
                try:
                    cfg = clip_config(parse_config(text), constraints)
                    cv, gen = evaluate_config(cfg, data, constraints, bbox_func, args.seed)
                    record.update({"status": "ok", "cv_score": cv, "generalization_score": gen})
                    for name in FEATURES:
                        record[name] = cfg[name]
                except Exception as exc:
                    record.update({"status": "failed", "error": repr(exc)})
                    record["cv_score"] = math.nan
                    record["generalization_score"] = math.nan
                rows.append(record)
                pd.DataFrame(rows).to_csv(out_dir / "direct_qwen_best_results.csv", index=False)
            request_idx += 1
            remaining -= batch_n

    results = pd.DataFrame(rows)
    summarize(results, out_dir)
    print("DIRECT_QWEN_BEST_TEST_COMPLETE")
    print(results[["sample_idx", "status", "cv_score", "generalization_score"] + FEATURES])


if __name__ == "__main__":
    asyncio.run(main())

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


TARGET_BINS = [
    ("0.6", 0.58, 0.63),
    ("0.7", 0.68, 0.73),
    ("0.8", 0.79, 0.83),
    ("0.9", 0.88, 0.905),
    ("0.92", 0.905, 0.925),
]


def load_history():
    frames = []
    qwen_path = Path("exp_bayesmark/results_discriminative/digits/RandomForest/0.csv")
    if qwen_path.exists():
        qwen = pd.read_csv(qwen_path)
        qwen["source"] = "qwen_original"
        qwen["cv_score"] = qwen["score"]
        frames.append(qwen)

    for path in sorted(Path("exp_baselines/results_single_compare/digits_RF_seed0").glob("*.csv")):
        df = pd.read_csv(path)
        if "cv_score" not in df:
            continue
        df["source"] = path.stem
        frames.append(df)

    cols = FEATURES + ["cv_score", "generalization_score", "source"]
    hist = pd.concat([f[cols] for f in frames], ignore_index=True)
    hist = hist.dropna(subset=["cv_score"])
    hist = hist.drop_duplicates(subset=FEATURES + ["cv_score"])
    return hist


def select_seed_points(history, per_bin):
    selected = []
    used = set()
    for label, low, high in TARGET_BINS:
        candidates = history[(history["cv_score"] >= low) & (history["cv_score"] < high)].copy()
        if candidates.empty:
            center = float(label)
            candidates = history.copy()
            candidates["dist"] = (candidates["cv_score"] - center).abs()
            candidates = candidates.sort_values("dist")
        else:
            center = (low + high) / 2
            candidates["dist"] = (candidates["cv_score"] - center).abs()
            candidates = candidates.sort_values("dist")

        count = 0
        for _, row in candidates.iterrows():
            key = tuple(round(float(row[f]), 8) for f in FEATURES)
            if key in used:
                continue
            used.add(key)
            item = row.to_dict()
            item["bin"] = label
            selected.append(item)
            count += 1
            if count >= per_bin:
                break
    return pd.DataFrame(selected)


def build_prompt(row, constraints):
    ranges = []
    for name in FEATURES:
        value_type, _, bounds = constraints[name]
        ranges.append(f"- {name}: [{bounds[0]}, {bounds[1]}] ({value_type})")
    current = ", ".join(f"{name}: {row[name]}" for name in FEATURES)
    return f"""You are improving RandomForest hyperparameters for a 10-class digits classification task.
The current configuration has measured CV accuracy {row['cv_score']:.6f}.

Current configuration:
## {current} ##

Allowed ranges:
{chr(10).join(ranges)}

Make a small local change that is likely to improve CV accuracy.
Rules:
- Stay inside the allowed ranges.
- Change at least one hyperparameter and at most three hyperparameters.
- Prefer small numerical changes, not a completely new global search.
- For RandomForest, low valid values for min_samples_leaf, min_samples_split, and min_weight_fraction_leaf may be useful.
- Do not copy the current configuration exactly.

Return only one configuration with this exact schema:
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
    missing = [f for f in FEATURES if f not in result]
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


def evaluate_config(cfg, data, hp_constraints, bbox_func, seed):
    np.random.seed(seed)
    random.seed(seed)
    fixed = dict(cfg)
    for name, value in fixed.items():
        if hp_constraints[name][0] == "int":
            fixed[name] = int(value)

    scorer = get_scorer("accuracy")
    model = bbox_func(**fixed)
    cv_score = float(np.mean(cross_val_score(model, data["train_x"], data["train_y"], scoring=scorer, cv=5)))

    model = bbox_func(**fixed)
    model.fit(data["train_x"], data["train_y"])
    gen_score = float(scorer(model, data["test_x"], data["test_y"]))
    return cv_score, gen_score


async def ask_llm(session, model, prompt, n, timeout):
    resp = await chat_completion(
        session=session,
        engine=model,
        messages=[
            {"role": "system", "content": "You are a careful hyperparameter tuning assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=160,
        top_p=0.95,
        n=n,
        request_timeout=timeout,
    )
    return [choice["message"]["content"] for choice in resp["choices"]]


def write_summary(out_dir, results):
    rows = []
    for bin_label, group in results.groupby("bin"):
        rows.append(
            {
                "bin": bin_label,
                "n": len(group),
                "mean_delta_cv": group["delta_cv"].mean(),
                "median_delta_cv": group["delta_cv"].median(),
                "success_rate": (group["delta_cv"] > 0).mean(),
                "best_delta_cv": group["delta_cv"].max(),
                "worst_delta_cv": group["delta_cv"].min(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("bin")
    summary.to_csv(out_dir / "summary_by_bin.csv", index=False)

    table_lines = ["| bin | n | mean_delta_cv | median_delta_cv | success_rate | best_delta_cv | worst_delta_cv |"]
    table_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        table_lines.append(
            "| {bin} | {n:d} | {mean_delta_cv:.6f} | {median_delta_cv:.6f} | "
            "{success_rate:.3f} | {best_delta_cv:.6f} | {worst_delta_cv:.6f} |".format(
                bin=row["bin"],
                n=int(row["n"]),
                mean_delta_cv=float(row["mean_delta_cv"]),
                median_delta_cv=float(row["median_delta_cv"]),
                success_rate=float(row["success_rate"]),
                best_delta_cv=float(row["best_delta_cv"]),
                worst_delta_cv=float(row["worst_delta_cv"]),
            )
        )

    lines = [
        "# LLM Local Hyperparameter Edit Test",
        "",
        "给 Qwen 一组已经真实评估过的 RandomForest 超参数，让它做小幅修改，然后真实重新评估修改前后的 CV/test accuracy。",
        "",
        "## Summary By Bin",
        "",
        "\n".join(table_lines),
        "",
        "## Overall",
        "",
        f"- Total edits: {len(results)}",
        f"- Success rate: {(results['delta_cv'] > 0).mean():.3f}",
        f"- Mean delta CV: {results['delta_cv'].mean():.6f}",
        f"- Median delta CV: {results['delta_cv'].median():.6f}",
        f"- Best delta CV: {results['delta_cv'].max():.6f}",
        f"- Worst delta CV: {results['delta_cv'].min():.6f}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="exp_bayesmark/results_llm_local_edit/digits_RF_seed0")
    parser.add_argument("--per-bin", type=int, default=3)
    parser.add_argument("--edits-per-point", type=int, default=2)
    parser.add_argument("--model", default=os.environ.get("LLAMBO_MODEL", "qwen3.5:9b"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("hp_configurations/bayesmark.json") as f:
        hp_constraints = json.load(f)["RandomForest"]
    with open("bayesmark/data/digits.pickle", "rb") as f:
        data = pickle.load(f)

    history = load_history()
    seeds = select_seed_points(history, args.per_bin)
    seeds.to_csv(out_dir / "seed_points.csv", index=False)

    bbox_func = get_bayesmark_func("RandomForest", "classification", data["test_y"])
    rows = []
    async with ClientSession(trust_env=True) as session:
        for seed_idx, row in seeds.reset_index(drop=True).iterrows():
            prompt = build_prompt(row, hp_constraints)
            texts = await ask_llm(session, args.model, prompt, args.edits_per_point, get_request_timeout(240))
            for edit_idx, text in enumerate(texts):
                record = {
                    "seed_idx": seed_idx,
                    "edit_idx": edit_idx,
                    "bin": row["bin"],
                    "source": row["source"],
                    "base_cv": float(row["cv_score"]),
                    "base_generalization": float(row["generalization_score"]),
                    "raw_response": text,
                }
                for name in FEATURES:
                    record[f"base_{name}"] = row[name]
                try:
                    cfg = clip_config(parse_config(text), hp_constraints)
                    changed = any(abs(float(cfg[name]) - float(row[name])) > 1e-12 for name in FEATURES)
                    if not changed:
                        raise ValueError("LLM copied the base configuration exactly")
                    cv, gen = evaluate_config(cfg, data, hp_constraints, bbox_func, args.seed)
                    record.update({"status": "ok", "new_cv": cv, "new_generalization": gen})
                    record["delta_cv"] = cv - float(row["cv_score"])
                    record["delta_generalization"] = gen - float(row["generalization_score"])
                    for name in FEATURES:
                        record[f"new_{name}"] = cfg[name]
                except Exception as exc:
                    record.update({"status": "failed", "error": repr(exc)})
                    record["new_cv"] = math.nan
                    record["new_generalization"] = math.nan
                    record["delta_cv"] = math.nan
                    record["delta_generalization"] = math.nan
                rows.append(record)
                pd.DataFrame(rows).to_csv(out_dir / "local_edit_results.csv", index=False)

    results = pd.DataFrame(rows)
    ok = results[results["status"] == "ok"].copy()
    if not ok.empty:
        write_summary(out_dir, ok)
    print("LOCAL_EDIT_TEST_COMPLETE")
    print(results[["bin", "source", "base_cv", "new_cv", "delta_cv", "status"]])


if __name__ == "__main__":
    asyncio.run(main())

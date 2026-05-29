import json
import os
import pickle
import random

import numpy as np
import pandas as pd
from bayesmark.bbox_utils import get_bayesmark_func
from llambo.llambo import LLAMBO
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


DATASET = "digits"
MODEL = "RandomForest"
SEED = 0
SAVE_DIR = "exp_bayesmark/results_discriminative_qwen_patch/digits/RandomForest"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(f"bayesmark/data/{DATASET}.pickle", "rb") as f:
        data = pickle.load(f)
    with open("hp_configurations/bayesmark.json", "r") as f:
        hp_constraints = json.load(f)[MODEL]

    task_context = {
        "model": MODEL,
        "task": "classification",
        "tot_feats": data["train_x"].shape[1],
        "cat_feats": 0,
        "num_feats": data["train_x"].shape[1],
        "n_classes": len(np.unique(data["train_y"])),
        "metric": "accuracy",
        "lower_is_better": False,
        "num_samples": data["train_x"].shape[0],
        "hyperparameter_constraints": hp_constraints,
    }

    bbox_func = get_bayesmark_func(MODEL, "classification", data["test_y"])

    def init_f(n_samples):
        return pd.read_json(f"bayesmark/configs/{MODEL}/{SEED}.json").head(n_samples).to_dict(orient="records")

    def evaluate_point(candidate_config):
        np.random.seed(SEED)
        random.seed(SEED)
        cfg = dict(candidate_config)
        for hyperparam, value in list(cfg.items()):
            if hp_constraints[hyperparam][0] == "int":
                cfg[hyperparam] = int(value)

        x_train, x_test = data["train_x"], data["test_x"]
        y_train, y_test = data["train_y"], data["test_y"]
        scorer = get_scorer("accuracy")

        model = bbox_func(**cfg)
        cv_score = float(np.mean(cross_val_score(model, x_train, y_train, scoring=scorer, cv=5)))

        model = bbox_func(**cfg)
        model.fit(x_train, y_train)
        generalization_score = float(scorer(model, x_test, y_test))
        return cfg, {"score": cv_score, "generalization_score": generalization_score}

    llambo = LLAMBO(
        task_context,
        "discriminative",
        n_candidates=10,
        n_templates=2,
        n_gens=10,
        alpha=0.1,
        n_initial_samples=5,
        n_trials=25,
        init_f=init_f,
        bbox_eval_f=evaluate_point,
        chat_engine="qwen3.5:9b",
        top_pct=None,
    )
    llambo.seed = SEED
    configs, fvals = llambo.optimize()

    search_history = pd.concat([configs, fvals], axis=1)
    search_history.to_csv(f"{SAVE_DIR}/{SEED}.csv", index=False)
    with open(f"{SAVE_DIR}/{SEED}_search_info.json", "w") as f:
        json.dump(
            {
                "llm_query_cost_breakdown": llambo.llm_query_cost,
                "llm_query_time_breakdown": llambo.llm_query_time,
                "llm_query_cost": sum(llambo.llm_query_cost),
                "llm_query_time": sum(llambo.llm_query_time),
            },
            f,
            indent=2,
        )
    print("PATCHED_QWEN_RUN_COMPLETE")
    print(search_history)


if __name__ == "__main__":
    main()

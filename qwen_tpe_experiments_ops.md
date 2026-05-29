# Qwen / TPE / Hybrid Experiments 操作文档

本文档记录最近新增的几组 LLAMBO 相关实验，目标是验证 Qwen3.5-9B 在 HPO 任务中的直接超参数先验、与 TPE 的差异，以及 Qwen warm-start + TPE 的混合路线是否更稳。

## 1. 实验环境

服务器：

```bash
ssh ts8
```

项目目录：

```bash
cd /home/doulingfeng/LLAMBO
```

Conda 环境：

```bash
source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh
conda activate llambo
```

本地 Ollama 模型：

```bash
ollama list
```

本轮实验使用：

```text
qwen3.5:9b
```

远程新增脚本位置：

```text
/home/doulingfeng/LLAMBO/exp_bayesmark/direct_qwen_best_hparams_test.py
/home/doulingfeng/LLAMBO/exp_bayesmark/direct_qwen_multi_task_test.py
/home/doulingfeng/LLAMBO/exp_bayesmark/tpe_multi_task_compare.py
/home/doulingfeng/LLAMBO/exp_bayesmark/hybrid_qwen_tpe_compare.py
```

本地备份脚本位置：

```text
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\direct_qwen_best_hparams_test.py
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\direct_qwen_multi_task_test.py
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\tpe_multi_task_compare.py
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\hybrid_qwen_tpe_compare.py
```

## 2. 新增实验一：直接让 Qwen 给 digits + RandomForest 最优超参

目的：不走 LLAMBO 的 surrogate / acquisition 流程，直接让 Qwen 根据任务描述输出一组它认为最好的 RandomForest 超参数，重复 30 次，然后真实训练评估。

运行命令：

```bash
cd /home/doulingfeng/LLAMBO
source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh
conda activate llambo
python exp_bayesmark/direct_qwen_best_hparams_test.py \
  --out-dir exp_bayesmark/results_direct_qwen_best/digits_RF_seed0 \
  --model qwen3.5:9b \
  --n 30 \
  --batch-size 5 \
  --seed 0
```

评估设置：

```text
Dataset: digits
Model: RandomForest
Metric: 5-fold CV accuracy
Test metric: held-out generalization accuracy
Seed: 0
Number of Qwen samples: 30
```

远程结果：

```text
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_best/digits_RF_seed0/direct_qwen_best_results.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_best/digits_RF_seed0/summary.json
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_best/digits_RF_seed0/prompt.txt
```

本地结果：

```text
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_best\direct_qwen_best_results.csv
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_best\summary.json
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_best\prompt.txt
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_best\direct_qwen_best_cv.svg
```

核心结果：

| 指标 | 数值 |
|---|---:|
| 请求次数 | 30 |
| 成功解析次数 | 30 |
| 解析成功率 | 1.000 |
| mean CV accuracy | 0.7676 |
| median CV accuracy | 0.8845 |
| best CV accuracy | 0.9005 |
| worst CV accuracy | 0.1058 |
| best generalization accuracy | 0.9111 |
| unique config count | 18 |

最好配置：

```json
{
  "max_depth": 15,
  "max_features": 0.5,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 0.01,
  "min_samples_split": 0.01,
  "min_weight_fraction_leaf": 0.01
}
```

结论：Qwen 直接给超参时有明显可用先验，最好结果能到 0.90 左右，但分布很宽，少数输出会非常差。这说明模型不是完全不懂 HPO，之前 LLAMBO 复现差的主要问题更可能在适配层、候选生成稳定性、surrogate 预测和解析约束上。

## 3. 新增实验二：10 个任务上直接让 Qwen 给超参

目的：把“直接 Qwen 给最佳超参”扩展到多个数据集、算法和任务，测试 Qwen 本身的上限、下限和期望。

默认 10 个任务：

| 序号 | Dataset | Model | Metric | 方向 |
|---:|---|---|---|---|
| 1 | digits | SVM | accuracy | 越大越好 |
| 2 | digits | MLP_SGD | accuracy | 越大越好 |
| 3 | wine | RandomForest | accuracy | 越大越好 |
| 4 | wine | SVM | accuracy | 越大越好 |
| 5 | wine | AdaBoost | accuracy | 越大越好 |
| 6 | breast | DecisionTree | accuracy | 越大越好 |
| 7 | breast | RandomForest | accuracy | 越大越好 |
| 8 | iris | MLP_SGD | accuracy | 越大越好 |
| 9 | diabetes | RandomForest | MSE | 越小越好 |
| 10 | diabetes | SVM | MSE | 越小越好 |

运行命令：

```bash
cd /home/doulingfeng/LLAMBO
source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh
conda activate llambo
python exp_bayesmark/direct_qwen_multi_task_test.py \
  --out-dir exp_bayesmark/results_direct_qwen_multitask \
  --model qwen3.5:9b \
  --n 30 \
  --batch-size 5 \
  --seed 0
```

如果只跑部分任务：

```bash
python exp_bayesmark/direct_qwen_multi_task_test.py \
  --out-dir exp_bayesmark/results_direct_qwen_multitask_subset \
  --model qwen3.5:9b \
  --n 30 \
  --seed 0 \
  --tasks digits:SVM diabetes:RandomForest
```

长任务后台运行示例：

```bash
mkdir -p exp_bayesmark/run_logs
nohup bash -lc 'cd /home/doulingfeng/LLAMBO && source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh && conda activate llambo && python exp_bayesmark/direct_qwen_multi_task_test.py --out-dir exp_bayesmark/results_direct_qwen_multitask --model qwen3.5:9b --n 30 --batch-size 5 --seed 0' > exp_bayesmark/run_logs/direct_qwen_multitask.log 2>&1 &
```

查看进度：

```bash
tail -f /home/doulingfeng/LLAMBO/exp_bayesmark/run_logs/direct_qwen_multitask.log
```

远程结果：

```text
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_multitask/summary_all_tasks.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_multitask/summary_all_tasks.json
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_multitask/<dataset>_<model>/results.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_multitask/<dataset>_<model>/summary.json
/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_multitask/<dataset>_<model>/prompt.txt
```

本地结果：

```text
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_multitask\summary_all_tasks.csv
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_multitask\summary_cn.md
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_multitask\multitask_summary.svg
```

Qwen 直接超参结果：

| Task | Metric | OK | Legal | Mean | Median | Best | Worst |
|---|---|---:|---:|---:|---:|---:|---:|
| digits + SVM | accuracy | 30/30 | 1.000 | 0.9898 | 0.9910 | 0.9916 | 0.9875 |
| digits + MLP_SGD | accuracy | 30/30 | 1.000 | 0.7776 | 0.8601 | 0.9311 | 0.1517 |
| wine + RandomForest | accuracy | 27/30 | 0.963 | 0.8917 | 0.9298 | 0.9722 | 0.3874 |
| wine + SVM | accuracy | 30/30 | 1.000 | 0.7799 | 0.7680 | 0.8313 | 0.7537 |
| wine + AdaBoost | accuracy | 30/30 | 1.000 | 0.9178 | 0.9010 | 0.9650 | 0.8938 |
| breast + DecisionTree | accuracy | 29/30 | 0.897 | 0.9019 | 0.9143 | 0.9231 | 0.8286 |
| breast + RandomForest | accuracy | 27/30 | 1.000 | 0.9144 | 0.9385 | 0.9516 | 0.6374 |
| iris + MLP_SGD | accuracy | 30/30 | 1.000 | 0.7042 | 0.7042 | 0.9417 | 0.3667 |
| diabetes + RandomForest | MSE | 25/30 | 1.000 | 0.5830 | 0.5491 | 0.5072 | 1.0044 |
| diabetes + SVM | MSE | 30/30 | 1.000 | 0.8639 | 0.9015 | 0.5020 | 1.0304 |

注意：diabetes 的指标是 MSE，越小越好；表中的 Best 对 diabetes 表示最低 MSE。

观察：

1. Qwen 在 digits + SVM、wine + AdaBoost、breast + RandomForest 等任务上直接给出的超参很强。
2. Qwen 在 MLP_SGD 和 diabetes + SVM 上波动明显，best 可以不错，但 median 或 worst 不稳定。
3. 树模型上存在参数名拼写错误，常见错误是把 `min_weight_fraction_leaf` 写成类似 `min_weight_fraction_fraction_leaf` 的形式。

## 4. 新增实验三：同样 10 个任务用 TPE 跑 30 次

目的：给 Qwen 直接超参实验一个传统 HPO baseline。这里使用 Optuna TPE，每个任务同样 30 次真实评估。

运行命令：

```bash
cd /home/doulingfeng/LLAMBO
source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh
conda activate llambo
python exp_bayesmark/tpe_multi_task_compare.py \
  --out-dir exp_bayesmark/results_tpe_multitask \
  --n-trials 30 \
  --n-startup-trials 5 \
  --seed 0
```

后台运行示例：

```bash
mkdir -p exp_bayesmark/run_logs
nohup bash -lc 'cd /home/doulingfeng/LLAMBO && source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh && conda activate llambo && python exp_bayesmark/tpe_multi_task_compare.py --out-dir exp_bayesmark/results_tpe_multitask --n-trials 30 --n-startup-trials 5 --seed 0' > exp_bayesmark/run_logs/tpe_multitask.log 2>&1 &
```

远程结果：

```text
/home/doulingfeng/LLAMBO/exp_bayesmark/results_tpe_multitask/summary_all_tasks.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_tpe_multitask/summary_all_tasks.json
/home/doulingfeng/LLAMBO/exp_bayesmark/results_tpe_multitask/<dataset>_<model>/results.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_tpe_multitask/<dataset>_<model>/summary.json
```

本地结果：

```text
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\tpe_multitask\summary_all_tasks.csv
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\tpe_multitask\qwen_vs_tpe_comparison.csv
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\tpe_multitask\qwen_vs_tpe_comparison.md
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\tpe_multitask\qwen_vs_tpe_best.svg
```

Qwen vs TPE 总结：

| Task | Metric | Qwen Best | TPE Best | Best Winner | Qwen Median | TPE Median | Median Winner |
|---|---|---:|---:|---|---:|---:|---|
| digits + SVM | accuracy | 0.9916 | 0.9923 | TPE | 0.9910 | 0.9910 | TPE |
| digits + MLP_SGD | accuracy | 0.9311 | 0.9603 | TPE | 0.8601 | 0.8956 | TPE |
| wine + RandomForest | accuracy | 0.9722 | 0.9441 | Qwen | 0.9298 | 0.8945 | Qwen |
| wine + SVM | accuracy | 0.8313 | 0.8313 | Tie | 0.7680 | 0.8102 | TPE |
| wine + AdaBoost | accuracy | 0.9650 | 0.9648 | Qwen | 0.9010 | 0.9190 | TPE |
| breast + DecisionTree | accuracy | 0.9231 | 0.9077 | Qwen | 0.9143 | 0.9077 | Qwen |
| breast + RandomForest | accuracy | 0.9516 | 0.9363 | Qwen | 0.9385 | 0.9187 | Qwen |
| iris + MLP_SGD | accuracy | 0.9417 | 0.9500 | TPE | 0.7042 | 0.7042 | Tie |
| diabetes + RandomForest | MSE | 0.5072 | 0.5646 | Qwen | 0.5491 | 0.6342 | Qwen |
| diabetes + SVM | MSE | 0.5020 | 0.5098 | Qwen | 0.9015 | 0.7559 | TPE |

Win count：

```text
Best-of-30: Qwen 6, TPE 3, Tie 1
Median-of-30: Qwen 4, TPE 5, Tie 1
```

解释：Qwen 的 best-of-30 很强，说明它有可用先验；但 TPE 的 median 更稳，说明传统优化器在持续探索和稳定性上仍有优势。

## 5. 新增实验四：混合路线 Qwen10 + TPE20

目的：测试 Qwen 负责提供初始先验，TPE 负责后续局部搜索的混合方案。总真实评估次数仍然是 30 次，保持与 Qwen-only 和 TPE-only 公平对比。

具体做法：

```text
1. 从 Qwen 直接超参结果里读取每个任务的候选。
2. 过滤解析失败、越界、不合法、重复配置。
3. 每个任务取前 10 个合法且唯一的 Qwen 配置作为 warm-start。
4. 把这 10 个点真实评估后加入 Optuna study。
5. TPE 继续跑剩下 20 次。
```

运行命令：

```bash
cd /home/doulingfeng/LLAMBO
source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh
conda activate llambo
python exp_bayesmark/hybrid_qwen_tpe_compare.py \
  --out-dir exp_bayesmark/results_hybrid_qwen10_tpe20 \
  --qwen-dir exp_bayesmark/results_direct_qwen_multitask \
  --total-trials 30 \
  --n-warm 10 \
  --seed 0
```

后台运行示例：

```bash
mkdir -p exp_bayesmark/run_logs
nohup bash -lc 'cd /home/doulingfeng/LLAMBO && source /home/doulingfeng/miniconda3/etc/profile.d/conda.sh && conda activate llambo && python exp_bayesmark/hybrid_qwen_tpe_compare.py --out-dir exp_bayesmark/results_hybrid_qwen10_tpe20 --qwen-dir exp_bayesmark/results_direct_qwen_multitask --total-trials 30 --n-warm 10 --seed 0' > exp_bayesmark/run_logs/hybrid_qwen10_tpe20.log 2>&1 &
```

远程结果：

```text
/home/doulingfeng/LLAMBO/exp_bayesmark/results_hybrid_qwen10_tpe20/summary_all_tasks.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_hybrid_qwen10_tpe20/summary_all_tasks.json
/home/doulingfeng/LLAMBO/exp_bayesmark/results_hybrid_qwen10_tpe20/<dataset>_<model>/results.csv
/home/doulingfeng/LLAMBO/exp_bayesmark/results_hybrid_qwen10_tpe20/<dataset>_<model>/summary.json
```

本地结果：

```text
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\hybrid_qwen10_tpe20\summary_all_tasks.csv
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\hybrid_qwen10_tpe20\three_way_comparison.csv
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\hybrid_qwen10_tpe20\three_way_comparison.md
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\hybrid_qwen10_tpe20\three_way_best.svg
```

三路对比：

| Task | Metric | Qwen Best | TPE Best | Hybrid Best | Best Winner | Qwen Median | TPE Median | Hybrid Median | Median Winner |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| digits + SVM | accuracy | 0.9916 | 0.9923 | 0.9930 | Hybrid | 0.9910 | 0.9910 | 0.9910 | TPE |
| digits + MLP_SGD | accuracy | 0.9311 | 0.9603 | 0.9590 | TPE | 0.8601 | 0.8956 | 0.8970 | Hybrid |
| wine + RandomForest | accuracy | 0.9722 | 0.9441 | 0.9650 | Qwen | 0.9298 | 0.8945 | 0.9298 | Qwen/Hybrid |
| wine + SVM | accuracy | 0.8313 | 0.8313 | 0.8313 | Qwen/TPE/Hybrid | 0.7680 | 0.8102 | 0.8099 | TPE |
| wine + AdaBoost | accuracy | 0.9650 | 0.9648 | 0.9650 | Qwen/Hybrid | 0.9010 | 0.9190 | 0.9154 | TPE |
| breast + DecisionTree | accuracy | 0.9231 | 0.9077 | 0.9231 | Qwen/Hybrid | 0.9143 | 0.9077 | 0.8923 | Qwen |
| breast + RandomForest | accuracy | 0.9516 | 0.9363 | 0.9516 | Qwen/Hybrid | 0.9385 | 0.9187 | 0.9165 | Qwen |
| iris + MLP_SGD | accuracy | 0.9417 | 0.9500 | 0.9500 | TPE/Hybrid | 0.7042 | 0.7042 | 0.7208 | Hybrid |
| diabetes + RandomForest | MSE | 0.5072 | 0.5646 | 0.5072 | Qwen/Hybrid | 0.5491 | 0.6342 | 0.5924 | Qwen |
| diabetes + SVM | MSE | 0.5020 | 0.5098 | 0.5020 | Qwen/Hybrid | 0.9015 | 0.7559 | 0.7088 | Hybrid |

Win count：

```text
Best-of-30:
- Hybrid: 1
- Qwen: 1
- Qwen/Hybrid: 5
- Qwen/TPE/Hybrid: 1
- TPE: 1
- TPE/Hybrid: 1

Median-of-30:
- Hybrid: 3
- Qwen: 3
- Qwen/Hybrid: 1
- TPE: 3
```

解释：混合路线不是每个任务都绝对更好，但它保留了 Qwen 的强先验，同时在 MLP 和 diabetes + SVM 这种 Qwen 方差较大的任务上提高了中位数稳定性。

## 6. 结果同步到本地

从服务器拉取结果到本地 PowerShell：

```powershell
scp -r ts8:/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_best/digits_RF_seed0 .\results\direct_qwen_best
scp -r ts8:/home/doulingfeng/LLAMBO/exp_bayesmark/results_direct_qwen_multitask .\results\direct_qwen_multitask
scp -r ts8:/home/doulingfeng/LLAMBO/exp_bayesmark/results_tpe_multitask .\results\tpe_multitask
scp -r ts8:/home/doulingfeng/LLAMBO/exp_bayesmark/results_hybrid_qwen10_tpe20 .\results\hybrid_qwen10_tpe20
```

当前本地汇总目录：

```text
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_best
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\direct_qwen_multitask
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\tpe_multitask
C:\Users\15808\Documents\Codex\2026-05-27\host-ts8-hostname-101-6-64\results\hybrid_qwen10_tpe20
```

## 7. 读结果时的注意事项

1. accuracy 任务越大越好，diabetes 的 MSE 越小越好。
2. `best_cv` 是交叉验证上的最优结果，不等于测试集泛化结果。
3. `generalization` 是最佳配置在 held-out test split 上的表现。
4. Qwen 实验里的 `parse_success_rate` 只表示能不能解析成配置，不代表配置一定合法或有效。
5. `raw_in_range_rate` 表示 Qwen 原始输出是否在搜索空间内；越界值在评估前会被修正或裁剪。
6. Hybrid 当前用的是“前 10 个合法唯一 Qwen 配置” warm-start，不是“Qwen 30 个里真实表现最好的 10 个”。
7. 所有这些实验目前都是 seed0，不能声称跨随机种子稳定，只能作为机制验证和方向判断。

## 8. 当前可以支持的判断

1. Qwen3.5-9B 不是完全不能用于 HPO。直接让它给超参时，多个任务 best-of-30 能接近或超过 TPE。
2. Qwen 的主要问题是稳定性和格式可靠性，不是完全没有任务先验。
3. TPE 的优势在于连续优化和中位数稳定性，尤其是 MLP_SGD 和部分 SVM 任务。
4. 混合路线是值得继续做的方向，因为它可以把 Qwen 的先验当作 warm-start，再让 TPE 做后续搜索。

## 9. 建议下一步实验

下一步建议做 warm-start 比例消融：

```text
Qwen5 + TPE25
Qwen10 + TPE20
Qwen15 + TPE15
Top-Qwen5 + TPE25
Top-Qwen10 + TPE20
```

其中 `Top-QwenK + TPE` 是更关键的版本：先让 Qwen 生成 30 个候选并真实评估，再用表现最好的 K 个作为 TPE warm-start。这能回答“LLM 是否适合做候选生成器，而不是完整优化器”这个问题。

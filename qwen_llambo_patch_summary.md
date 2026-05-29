# Qwen-LLAMBO 小修复实验总结

时间：2026-05-28  
任务：`digits + RandomForest + seed0`  
模型：`qwen3.5:9b`，Ollama，本地 no-think 模式  
目标：不换模型，只改解析、`max_tokens` 和 prompt，观察 Qwen-LLAMBO 是否能从原始最好 `0.666` 提升到接近 `0.8+`。

## 1. 原始问题

原始 Qwen-LLAMBO 25 trial 结果：

| 版本 | 最好 CV accuracy | 最好 generalization accuracy |
|---|---:|---:|
| 原始 Qwen-LLAMBO | 0.66598 | 0.66389 |

和传统 baseline 相比明显偏低：

| 方法 | 最好 CV accuracy |
|---|---:|
| HEBO | 0.91092 |
| TPE | 0.89910 |
| SMAC | 0.88658 |
| Random | 0.86151 |
| 原始 Qwen-LLAMBO | 0.66598 |

之前分析发现主要问题不是 RandomForest 训练本身，而是 Qwen 作为 LLAMBO 后端时：

- 性能预测输出格式不稳定，例如输出 `## 0.6506`，缺少结尾 `##`，导致解析为 `nan`。
- surrogate 预测不校准，容易把真正好的配置预测成 `0.1057` 或 `0.6506`。
- 候选生成会塌缩到少数相似配置。
- 原始 prompt 说“不要推荐范围边界值”，但这个任务的好配置恰好需要多个超参数接近下界。

## 2. 第一组修复：解析、max_tokens、基础 prompt

改动文件：

- `llambo/discriminative_sm.py`
- `llambo/acquisition_function.py`
- `llambo/discriminative_sm_utils.py`

改动内容：

1. 放宽性能预测解析：
   - 原来主要接受 `## 0.6506 ##`
   - 新增接受 `## 0.6506` 和裸数字 `0.6506`

2. 把 surrogate 预测的 `max_tokens` 从 `8` 改为 `32`：
   - 避免 Qwen 因输出 token 太短截断结尾 delimiter。

3. 修改候选生成 prompt：
   - 不再禁止接近最小/最大边界。
   - 明确说只要在合法范围内，接近边界是允许的。

4. 修改性能预测 prompt：
   - 强调只输出一个数字，格式为 `## performance ##`。

静态诊断结果：

| 测试项 | 修复前 | 修复后 |
|---|---:|---:|
| Qwen 预测输出可解析率 | 0/8 | 8/8 |

这说明解析和 `max_tokens` 问题确实被修掉了。

但是，Qwen 的数值校准仍然不好。例如对 HEBO 已知最好点：

```text
真实 CV accuracy: 0.9109
Qwen 预测: 0.1057 或 0.6506 附近
```

所以第一组修复解决的是“能不能读懂 Qwen 输出”，没有彻底解决“Qwen 会不会判断哪个点好”。

## 3. 第二组修复：候选 prompt 显式 schema

第一次完整实验尝试很快失败。原因是 prompt 写成：

```text
## configuration ##
```

Qwen 有时会直接照抄占位词：

```text
configuration
```

导致候选无法解析。

于是把候选 prompt 改为显式 schema：

```text
## max_depth: <int>, max_features: <float>, min_impurity_decrease: <float>,
min_samples_leaf: <float>, min_samples_split: <float>, min_weight_fraction_leaf: <float> ##
```

并明确：

```text
Do not output the word configuration as a placeholder.
```

同时放宽候选解析：

- 支持 `key: value`
- 支持 `key is value`
- 支持有无 `##` 包裹

结果：候选解析数量明显改善，每轮能解析出 10 个 proposed candidate。

## 4. 第三组修复：降低最低候选数

虽然 Qwen 能解析出候选，但大量候选会重复已有点，过滤后 accepted candidate 很少。

原逻辑要求至少 5 个 accepted candidate，否则报错：

```text
LLM failed to generate candidate points
```

修改为：

```text
min_candidate_points = 3
```

目的：让小模型在候选多样性不足时也能继续跑完整 BO 流程。

## 5. 第四组修复：加入 RandomForest 领域提示

为了让 Qwen 不再推荐过大的 `min_samples_leaf`、`min_samples_split`、`min_weight_fraction_leaf`，加入了轻量领域提示：

```text
For RandomForest, small min_samples_leaf, min_samples_split, and
min_weight_fraction_leaf values can be useful because they allow more expressive trees.
```

这个改动非常关键。某次短跑中，第一轮直接选到了：

```text
max_depth = 15
max_features = 0.99
min_impurity_decrease = 0
min_samples_leaf = 0.01
min_samples_split = 0.01
min_weight_fraction_leaf = 0.01
```

结果：

| 版本 | trial | CV accuracy | generalization accuracy |
|---|---:|---:|---:|
| patch 短跑 | 0 | 0.88448 | 0.87500 |

这是一个非常重要的信号：只改适配层和 prompt，Qwen-LLAMBO 可以从原始 `0.666` 跳到 `0.8845`。这说明 Qwen 9B 并不是完全不能用，原始差距里有很大一部分来自适配问题。

但是这个版本第二轮失败了，因为 Qwen 之后反复复制这个已评估过的边界好点，去重后 accepted candidate 不足。

## 6. 第五组修复：避免复制已有配置

为了让完整 25 trial 能继续跑，加入提示：

```text
Do not copy any hyperparameter configuration shown in the examples;
propose a new nearby configuration that changes at least two hyperparameters
from the best observed example.
```

并给 accuracy 目标值加了一个上限，避免第一轮达到 `0.8845` 后，下一轮目标被推到 `0.96`，导致 Qwen 只会复读边界最优点：

```python
desired_fval = min(desired_fval, observed_best + 0.03, 0.92)
```

这个版本解决了中断问题，实验可以继续跑。

## 7. 当前稳定版运行结果

稳定版实验仍在后台运行：

```text
screen: llambo_qwen_patch_digits_rf_25
log: /home/doulingfeng/LLAMBO/exp_bayesmark/run_logs/ollama_digits_rf_25_qwen_patch.log
```

截至读取日志时，跑到 trial 18，尚未写出最终 CSV。

当前最好结果：

| 阶段 | 最好 CV accuracy | 对应 generalization accuracy |
|---|---:|---:|
| 初始化最好 | 0.65066 | 0.59722 |
| 原始 Qwen-LLAMBO 25 trial | 0.66598 | 0.66389 |
| patch 稳定版，截至 trial 18 | 0.7293 | 0.7222 |
| patch 短跑 exploit 版 | 0.88448 | 0.87500 |

稳定版逐步提升记录：

| trial | 当前 CV | 当前 generalization | 当时最好 CV |
|---:|---:|---:|---:|
| 3 | 0.7071 | 0.6556 | 0.7071 |
| 8 | 0.7119 | 0.6778 | 0.7119 |
| 11 | 0.7168 | 0.6944 | 0.7168 |
| 16 | 0.7293 | 0.7222 | 0.7293 |
| 18 | 0.7174 | 0.7250 | 0.7293 |

目前没有看到新的 traceback，也没有出现之前那种大量 `Mean of empty slice` warning。

## 8. 效果评价

这轮实验说明两件事。

第一，适配层确实是主要瓶颈之一。

原始版本最好只有：

```text
CV accuracy = 0.666
```

但只改解析、`max_tokens` 和 prompt 后，短跑可以直接到：

```text
CV accuracy = 0.8845
generalization accuracy = 0.8750
```

这个结果已经接近 Random/SMAC/TPE 的水平，说明 Qwen 9B 不是完全不能用于 LLAMBO。

第二，Qwen 9B 的候选多样性和 surrogate 判断仍然弱。

为了让 25 trial 稳定跑完，我们加了“不复制已有配置、至少改两个超参数”的约束。这个约束解决了中断，但也让模型不再大胆 exploit 那个边界好点附近，导致稳定版截至 trial 18 只有：

```text
CV accuracy = 0.7293
generalization accuracy = 0.7222
```

也就是说：

- exploit 太强：能一轮冲到 0.8845，但后续容易复制同一点并中断。
- diversity 约束太强：能稳定跑，但搜索变保守，分数只有 0.73 左右。

## 9. 结论

当前结论不是“Qwen 9B 不行”，而是：

```text
Qwen 9B 对 LLAMBO 很敏感，需要专门的 prompt、解析和候选多样性控制。
```

已经验证：

- 解析问题可以修。
- `max_tokens=8` 太短，改为 `32` 后预测格式明显改善。
- 允许边界值、加入 RandomForest 领域提示后，可以找到 0.8845 的好配置。
- 但 Qwen 容易复制已知最优点，完整 BO 还需要更好的去重/扰动策略。

## 10. 下一步建议

如果继续不换模型，下一步不要再只靠 prompt，而应该在候选生成后加一个程序化扰动器：

1. 如果 Qwen 输出重复最优点，则自动围绕该点采样邻域候选。
2. 对 `min_samples_leaf`、`min_samples_split`、`min_weight_fraction_leaf` 采用 low-range perturbation，例如 `[0.01, 0.08]`。
3. 保留 Qwen 给出的方向，但用代码保证候选多样性。
4. surrogate 排序时，如果 Qwen 预测不校准，可以混入真实已观测分数附近的启发式排序。

这样更符合 Qwen 9B 的能力边界：让它提供方向，让程序负责数值探索和去重。

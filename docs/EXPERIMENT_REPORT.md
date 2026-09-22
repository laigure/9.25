# 实验报告

## 最终双目标 Grounding

200 epoch 训练完整结束，确定性 val 评测覆盖 1,300 个 pair：

| split / metric | 逐目标 Acc@0.25 | 逐目标 Acc@0.50 | 双目标 Joint@0.25 | 双目标 Joint@0.50 | mIoU |
|---|---:|---:|---:|---:|---:|
| train | 98.64% | - | 97.27% | - | 63.63% |
| val | 59.38% | 37.00% | 40.38% | 18.54% | 34.65% |

train/val 差距表明 1,137 个训练 pair 上存在明显过拟合。epoch 150 到 200 的 val 基本不再提升：逐目标 Acc@0.25 从 59.35% 到 59.38%，Joint@0.25 从 40.46% 到 40.38%，mIoU 从 34.64% 到 34.65%。继续增加相同训练轮数不是当前的主要解法。

## QA v4：Grounding 正确条件下

QA 输入仅使用问题和 contrastive 288D token。GT IoU≥0.25 仅用于选出这个诊断子集。

| 方法 | 总体 | relative location | which is left | closer to ego |
|---|---:|---:|---:|---:|
| relation Projector，LLM 冻结 | 78.86% | 63.58% | 92.50% | 100.00% |
| relation Projector + LoRA | **80.00%** | **65.74%** | **92.50%** | **100.00%** |
| 同类别 donor token 打乱 | 36.91% | 22.22% | 50.63% | 56.49% |
| A/B swap eval | 80.33% | 65.74% | 93.75% | 100.00% |

A/B swap 前后的预测逻辑一致率为 91.87% (565/615)。正常 token 与打乱 token 差距表明模型实际使用了对象 token，而不只是记忆问题模板和答案分布。LoRA 相对 Projector-only 只增加 1.14 个百分点，当前主要空间关系能力来自 token 交互与 Projector。

## 端到端 QA：不使用 IoU 筛选

| metric | 结果 |
|---|---:|
| QA 行数 | 1,638 |
| 可对齐 Grounding pair | 508 |
| 总准确率 | **57.08%** (935/1,638) |
| 加权多数类基线 | 35.29% |
| relative location | 37.23% |
| which is left | 73.65% |
| closer to ego | 99.60% |

按 Grounding 成功与否分组：

| Grounding 状态 | QA 行数 | QA 准确率 |
|---|---:|---:|
| 两个目标均 IoU≥0.25 | 615 | 80.00% |
| 至少一个目标未通过 | 1,023 | 43.30% |

Grounding 错误子集中，`relative_location` 下降到 21.83%，说明精确的二物体相对关系是当前端到端瓶颈。`closer_to_ego` 仍为 99.17%，说明即使预测框 IoU 低，token 中仍可保留很强的径向位置线索。

## 已排除的错误路线

1. **任意配对独立单目标记录**：早期重建集中很多目标在当前点云中不可见，导致低准确率。已改为只使用真实 `others` 关联。
2. **用 soft selector 的框筛选却接 contrastive token**：会让 token 与框不是同一 query。v4 已强制 selector 一致。
3. **只有独立 MLP，没有 A/B 关系交互**：旧版 QA 为 62.26%，`which_is_left` 为 56.10%。加入对称数据和 relation Transformer 后分别为 80.00% 和 92.50%。
4. **只报告通过 Grounding 的样本**：80.00% 是 token 能力上限诊断，不是完整系统指标。完整链路当前是 57.08%。
5. **继续堆 LoRA epoch**：最佳 LoRA 在 epoch 1，epoch 2 val loss 上升，已有过拟合信号。

## 下一步

1. 直接从 1,300 个 val multi-grounding pair 生成同结构 QA，消除当前只覆盖 508 pair 的对齐缺口。
2. 减少 Grounding 过拟合，重点提高 Joint@0.25 和 Joint@0.50，而不是只看逐目标指标。
3. 获得有独立 caption 的三目标及以上标注后，再训练可变目标数模型并报告 multi-target joint metric。
4. 将 QA 从固定空间关系扩展到障碍物识别、可通行区域和路径规划，同时保持规划标签由 GT 几何/轨迹生成。

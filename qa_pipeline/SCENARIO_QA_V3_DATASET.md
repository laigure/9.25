# 3EED 隐式 token 空间问答数据集 v3

v3 在 v2 的同一份中性 Grounding token 上增加**不同空间判断任务**。模型输入仍是自然语言问题与 1–3 个 288 维目标 token，经共享 projector 输入 Qwen；GT box 仅在离线生成答案和评测时使用。v2 数据与模型完整保留。

## 数量与独立性

| 划分 | QA 条数 | 独立帧 | 独立目标 |
|---|---:|---:|---:|
| 训练 | 8,902 | 2,250 | 2,957 |
| 验证 | 5,153 | 1,181 | 1,590 |
| 测试 | 4,219 | 1,088 | 1,414 |
| 合计实际入题 | 18,274 | 4,519 | 5,961 |

训练/验证/测试仍按 94/46/47 个完整驾驶序列划分，序列之间没有交集。18,274 是问答记录数，**不是 18,274 个不同场景**；多个问题可能来自同一帧或目标。原始中性 token bank 有 4,681 帧、6,130 个目标，其中 4,519 帧、5,961 个目标至少进入一条 QA。原始 bank 中含两个以上中性目标的帧为 1,228 个，其中三目标帧 186 个。

## 题型

| 题型 | 任务 | 训练条数 |
|---|---|---:|
| `ego_location` | 目标相对自车的联合方位 | 2,629 |
| `ego_side` | 目标在自车左侧还是右侧 | 1,957 |
| `ego_near_far` | 目标离自车近还是远；15m/30m 间隔剔除模糊样本 | 1,490 |
| `relative_location` | 两目标相对方位，包含反向提问 | 1,310 |
| `which_is_left` | 两个目标谁更靠左 | 694 |
| `closer_to_ego` | 两个目标谁离自车更近 | 509 |
| `closer_of_two` | 两个候选目标谁离第三个参照物更近 | 245 |
| `closest_of_three_to_ego` | 三个目标谁离自车最近 | 68 |

新题型使用 GT 几何规则制标，但公开 QA 的问题文本、token 引用和模型输入中都没有坐标、box、距离数值或预计算关系标签。描述沿用 v2 的中性指代。所有新近远/比较题都有边界间隔；测试集没有按预测 IoU 过滤。

## 产物与命令

- [公开 QA](/D:/大三上/asc/3eedQA/qa_pipeline/artifacts/qa_v3/scenario_qa/qa.jsonl)
- [离线 GT 标签](/D:/大三上/asc/3eedQA/qa_pipeline/artifacts/qa_v3/scenario_qa/private_gt.jsonl)
- [数量统计](/D:/大三上/asc/3eedQA/qa_pipeline/artifacts/qa_v3/scenario_qa/summary.json)
- [构建脚本](/D:/大三上/asc/3eedQA/qa_pipeline/build_scenario_qa_v3.py)

远程路径为 `/root/autodl-tmp/3eed_data/qa_v3/scenario_qa/`。训练使用 `/root/autodl-tmp/3eed_data/qa_v2/tokens/waymo_tokens.npz`，不需要重新导出 Grounding 特征。训练状态与结果持续记录在 [WORK_LOG.md](/D:/大三上/asc/3eedQA/qa_pipeline/WORK_LOG.md)。

现有三目标真值仍稀少，路线规划也仍缺未来轨迹与地图监督。增加问答类型可以提供更多监督，但无法代替新场景采集或更高质量的 Grounding token。

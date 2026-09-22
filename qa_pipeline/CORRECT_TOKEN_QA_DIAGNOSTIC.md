# 正确定位 token 接入 QA：条件诊断（2026-09-21）

## 目的与口径

检验现有 QA 模型在 Grounding 已选对所有目标时，是否能利用联合模型的隐式 288 维 token。联合 Grounding 仍按正常 soft-token 分数从候选 query 中选框；**GT 框只在选框后离线计算 IoU≥0.25，用于筛选诊断样本**。没有用 GT 从 256 个候选 query 中挑框，也没有向 QA 输入坐标、框或 GT 关系。QA 答案/评分沿用原 v3 私有 GT。

同一批问题、同一个冻结的 QA checkpoint（`qa_v3/runs/full_lora/best.pt`，projector 和 LoRA 均加载）比较三种 token：

1. **原单目标 token**：原 QA v3 输入。
2. **联合正确 token**：本轮多目标 3EED 从同一句选择并导出的有序 token，按 `object_id` 对回 A/B/C。
3. **联合 token 打乱**：给每个角色换成同 test split、同类别、不同物体的 token；题目和答案不变。

## 样本流失

联合 Grounding test 397 组中 23 组所有目标定位正确（5.79%）；原 QA test 的 1,276 道多目标题中，89 道对应这 23 组。三条件严格使用相同 89 个 `qa_id`，题目和答案逐字相同。题目保留率 **89/1276=6.97%**。同一 Grounding 组可生成多道 QA 题，因此 89 道题不是 89 个独立场景。

## 冻结 QA 模型结果

| 输入 token | 答对/89 | 准确率 |
|---|---:|---:|
| 原单目标 token | 43 | 48.31% |
| 联合正确 token | 44 | 49.44% |
| 联合 token 打乱 | 45 | 50.56% |

联合 vs 原 token：24/89 道题的生成文字发生变化，净增 1 题。联合 vs 打乱：12/89 道题的生成文字变化；打乱后 2 题由对变错、3 题由错变对，净增 1 题。相对位置题 42 道：原 12/42、联合 10/42、打乱 11/42。其余题型样本数较小，完整逐题结果见 `artifacts/multi_grounding/qa_correct_only/`。

## 解释与限制

**结论：现有冻结 QA checkpoint 在这批正确定位的联合 token 上没有可测收益；打乱 token 后成绩也没有下降。** 这不能证明联合 token 不含空间信息：checkpoint 原本在旧单目标 token 上训练，联合 token 分布有变化；89 道 QA 只来自 23 个成功定位的组合，题型和场景分布偏窄。筛选使用 GT，只能报告“定位正确条件下的 QA 诊断”，不能把 49.44% 当端到端 QA 成绩。若把定位失败样本也纳入端到端评价，需要对全量 1,276 道多目标题逐一推理和计分。

## 产物与下一步

- 构建脚本：`build_correct_multi_qa.py`；评测脚本：`run_correct_multi_qa.py`。
- 本地摘要与逐题结果：`artifacts/multi_grounding/qa_correct_only/`；远端三套完整 QA 输入：`/root/autodl-tmp/3eed_data/multi_grounding/qa_correct_only/{original,joint,shuffled}/`。
- 下一步先提高联合 Grounding 的全目标通过率，增加可用于 QA 的独立场景，再分别测冻结 QA 与适配联合 token 后的 QA。适配训练应只用 train split 的正确定位样本，val 选模型，test 保持未见。

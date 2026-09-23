# QA 输入泄漏审计

## 发现的问题

`scene_reasoning_qa_v1/qa.jsonl` 把每个目标的原始 3EED caption 拼进了 `question`。这些 caption 经常包含 `lower left`、`right`、`behind`、`in front of` 等位置描述。全部 5,518 条问题至少命中一个这类空间提示，因此大模型可能直接从文字推断标签。

此外，旧生成器在两个任务里通过角色顺序泄漏答案：

- `rank_then_relation` 先把第二近目标放到 Object A、最近目标放到 Object B。
- `local_path_planning` 先把最近障碍物放到 Object A。

因此 v1 只能证明数据生成和评测代码能运行，不能证明模型从 Grounding token 学会了空间关系。

## v2 的实际模型输入

`scene_reasoning_qa_v2_noleak` 的 LLM 输入由两部分组成：

1. 通用问题文字，例如 `Grounding tokens are provided for Objects A, B, C. Which object is second closest ...?`
2. 按角色排列的 288 维 Grounding token，例如 `[token_A, token_B, token_C]`。

`object_id` 是数据文件中的 token 对齐键，`ScenarioTrainer.pieces()` 不会将它编码成文字。答案字段只在 SFT 时作为监督目标接在 assistant 端，不属于用户问题输入。

原始 referring expression、物体类别、GT box/center、关系标签和规划中间量均保存在 `private_gt.jsonl`。它们负责离线生成答案和评测，不进入模型 prompt。

## 位置如何判断

3EED decoder query 在候选中心和尺寸的位置嵌入参与交互后产生每个目标的 288 维 token。关系 Projector 对 A–E 的 token 及角色 embedding 做一次关系 Transformer，再映射到 LLM hidden dimension。LLM 应根据这些连续向量之间的差异生成 left/right/front/behind、距离排序和局部避障答案。

这套设计没有向 LLM 提供可读坐标。它依赖 token 是否真的保留了足够的三维位置特征，所以最终必须通过正确 token、同类别打乱 token、无 token以及位置探针进行比较。

## v2 自动检查

- QA 总数：5,518，train 2,656，val 2,862。
- 原始 caption 出现在公开问题：0。
- 公开 `object_refs` 出现 description/category/bbox/center：0。
- 公开问题出现坐标格式：0。
- 公开问题出现 left/right/front/behind 答案提示词：0。
- 排序题正确角色覆盖 A–E，不再恒为 A。
- 规划题最近障碍物角色覆盖 A–E，不再恒为 A。
- canonical answer 回灌严格评测：关键词、三元组、完整句、一句话、规划动作、A/B 交换和严格合取均为 100%。

## 仍需控制的隐式泄漏

最终 decoder token 是语言条件化的：Grounding 模型为了选目标会读取 referring expression，因此 caption 中的空间词可能被编码进 288 维 token。v2 已消除 LLM prompt 的直接文本泄漏，但不能仅凭这一点证明 token 的答案来自点云几何。

正式实验至少需要比较：

1. 当前语言条件 decoder token。
2. 去除关系词后的中性 referring expression token。
3. 在同一目标 query 索引处提取的视觉或候选特征。
4. 同类别打乱 token 与无 token 基线。

只有正确 token 明显优于这些对照，且 Grounding 错误子集与正确子集符合预期，才能把结果解释为三维空间推理能力。

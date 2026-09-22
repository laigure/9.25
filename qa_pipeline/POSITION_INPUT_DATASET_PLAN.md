# Grounding token 的位置信息与 QA 数据集设计决策

## 已知事实，不把猜测当结论

- 当前公开 3EED Grounding 代码使用 `loc_learned`：每层 decoder 都把候选中心和尺寸 `[cx,cy,cz,sx,sy,sz]` 作为位置输入，经注意力交互；最终导出的 `last_query_features` 是 288 维、与文本交互后的候选 query。它**可能含**位置，但不存在“其中第几维就是 x/y/距离”的保证。最终预测框由后续预测头计算，框的数值与 288 维特征是两个不同输出。
- 当前导出脚本同时保存了选中候选的 288 维 token 与预测 center/size；QA v3 只把 token 输入 LLM。语言选 query 的 top1 可靠性仍不足，尤其双/三目标。
- 现有 token 线性探针在 v3 val 上：自车远近 89.37%、自车左右 69.42%、自车联合方位 52.84%、双物体相对位置 30.09%。这证明部分空间信息可从 token 读出，但不证明 LLM 能稳定利用它，也不保证选中的目标正确。
- 早期双物体 QA 测试集 332 题：仅 token 的联合方位 59.94%；额外输入**预测中心** 68.37%；额外输入预测框中心和尺寸 67.17%；使用**预测中心算出的相对向量/距离**诊断为 79.82%。最后一种几乎把答案所需的关系先算好，只能作为诊断上限，不应冒充模型自主空间推理。

## 关键区分

1. **标签真值**：GT object/box/center 用于离线自动生成正确答案和划分难度；无论采用什么模型输入，标签真值都不变。
2. **推理时的感知结果**：Grounding 的 token、预测 center/box、置信度、候选排序。这些可能有错，不能写回 GT 答案。
3. **LLM 实际输入**：可以只给 token，也可以把**预测**位置经过位置编码/MLP 变成向量与 token 融合。后者虽然不在提示词中出现数字，但在信息论上仍然是给了模型位置；需要在方法中如实说明，不能称为“仅 Grounding token”。

## 推荐的稳定数据结构

保持三层，避免把一个训练输入方案写死进数据集：

```text
Scene: sequence_id, frame_id, scene_id, point_cloud_ref, coordinate_frame, ego_pose_ref, split
Object GT (离线私有): scene_id, object_id, category, GT box/center/yaw, referring_expression
QA: qa_id, scene_id, scenario, question, answer, role_to_object_id, label_rule, difficulty, split
Grounding prediction (按 checkpoint 版本): scene_id, expression_id, candidate_id,
    query_feature, predicted_center/size, language_score, checkpoint_id, selection_policy
```

`QA` 用稳定的 `object_id` 引用目标，不直接写某个 token 索引。训练 loader 通过 `object_id → prediction` 联接，从而同一套 GT 问答可以比较师兄模型、公开模型和不同位置输入。GT 几何保存在离线标签表；LLM 只能通过独立的输入适配器看到选定的预测信息。

## 三种输入实验，用同一批 QA 和同一拆分

| 版本 | 每个目标给下游什么 | 用途 |
|---|---|---|
| A | 选中的 Grounding query token | 基础对照：token 自己能提供多少空间信息。 |
| B，推荐主线 | Grounding query token + 从**预测中心**生成的可学习/周期位置编码；在 projector 前融合或形成一个位置 soft token。提示词不出现坐标。 | 保留目标外观/指代与位置的独立通道；不使用 GT 位置。标注为“预测位置编码”，不可称为纯隐式。 |
| C | 另加预测目标之间的相对位置编码 | 诊断或后续扩展。若直接给 dx/dy/距离，模型几乎已经拿到关系答案，应与 B 分开报告。 |

B 的位置输入宜先从预测中心 `cx,cy,cz` 开始，统一自车坐标系和单位；可比较归一化 xyz 经过 MLP 与 Fourier/sine 编码经过 MLP。不要一开始把预测框尺寸、GT、预计算关系全部堆进去。绝对位置编码可以放在 Grounding 输出侧或 projector 侧，但必须记录来源与层位。对于师兄新模型，重新做 token 探针，不能沿用旧模型探针结论。

## 先验证再扩数据

1. Grounding：完整 val 上报告 top1 IoU/Acc@0.25、top-k 召回，以及同帧两个目标同时选中的比例。候选是否正确要与特征是否有空间信号分开测。
2. 表示：在固定正确候选与真实预测候选两套条件下，用冻结 token 的简单线性/小 MLP 探针预测左右、前后、远近和双物体关系；验证集按完整驾驶序列划分。探针只用于审计，不作为最终 LLM 输入。
3. QA：固定同一批问题/答案和训练预算，对比 A、B、C；除了总正确率，单独报告双目标关系、距离误差和按 Grounding 选对/选错分组的结果。输入加位置后若双目标仍弱，考虑关系模块或更直接的监督，而非继续增大 QA 文本量。

## 场景规划

- 现在的 3EED QA 先保留单目标方位、双目标相对位置、近远和候选比较；每条记录保留场景 ID、目标 ID、题型和难度。QA 答案始终由 GT 算出。
- 路线规划是另一层任务，需要未来轨迹、道路/车道、可行驶区域与动态障碍真值。当前空间 QA 不应把没有这些真值的“建议路线”当作可训练答案。

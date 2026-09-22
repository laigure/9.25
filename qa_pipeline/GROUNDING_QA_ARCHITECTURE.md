# Grounding → 隐式 token → 空间 QA：与第一阶段训练的接口

## 分工与整体链路

第一阶段先复现并测试 **3EED 论文公开 Grounding 模型**，明确输出特征、候选目标与语言指代如何匹配。师兄提出这项训练任务；当前由我们在远程 4090 上启动基线复现。第二阶段用冻结的 Grounding 输出训练 token 选择/聚合模块、projector 和 Qwen LoRA，并用同一套 GT 场景标注制作及评测空间 QA。

```text
点云 + 中性目标描述
    ↓
[阶段一：复现 3EED Grounding]
点云编码 → 语言编码与交互 → 候选目标 query → 目标选择/排序
    ↓  每个指代目标的隐式候选特征 [K, D]
[阶段二：QA，冻结 Grounding]
候选选择/聚合 → projector → Qwen LoRA + Question → 自然语言答案

GT object/box → 离线制 QA 答案、训练选择器监督、评测；不输入 LLM
```

第一阶段训练与第二阶段训练应分开报告。已下载的公开 3EED checkpoint、正在从头复现的 3EED 模型和 QA v3 是三个不同产物；此次**不是**训练师兄自己的新模型。

## 当前公开 3EED checkpoint 的实际结构

以下是核对当前仓库 `models/bdetr.py`、`models/point_backbone_module_v2.py` 和公开 checkpoint 配置后的结果；从头训练的模型结构沿用这套代码，最终效果需训练后验证。

| 模块 | 当前结构 / 输出 |
|---|---|
| 点云编码 | `Point_Backbone_V2`，基于 PointNet++ 的分层采样与特征传播；checkpoint 使用 RGB，输出约 1,024 个 288 维点特征与对应内部 xyz。 |
| 语言编码 | RoBERTa 编码，再投影到 288 维；RoBERTa 参数冻结。 |
| 跨模态编码 | 3 层双向 point/text cross-encoder。当前 checkpoint 的 BUTD box stream 为关闭状态。 |
| query 初始化 | 从点特征预测 objectness，取 256 个 query，先产生 proposal。 |
| decoder | 6 层 288 维双模态 decoder；内部可用 xyz/box 位置嵌入；最后一层状态与最终预测框索引一一对应。 |
| 输出与匹配 | 每个 query 对应一个候选 3D 框；64 维对比投影用于文本-query 排序。`last_query_features` 是用于 QA 的 288 维特征，不是 64 维匹配向量。 |

当前实际链路由 `export_neutral_tokens.py` 从 256 个 query 中按文本相似度取 top1，再将对应的 `last_query_features` 输入共享 MLP projector（288→1024→Qwen hidden size），后接 Qwen2.5-7B-Instruct 的 LoRA。QA 模型收到问题与 1–3 个隐式目标 token；box、中心、距离等数值不进入 LLM。

## 已发现的关键接口风险

当前 top1 token 的**层位正确，但目标选择不可靠**。全量 Waymo val 的 top1 Grounding IoU≥0.25 为 47.5%；v3 val 中双目标问题的全部输入目标同时达标仅 9.5%，三目标为 0/233。32 条 val 小样本的 GT 离线上限：top1 53.1%、文字 top5 中存在合格框 81.3%、全部 256 候选中存在合格框 96.9%。说明选错 query 是主要瓶颈之一；这些 top5/oracle 数字是离线诊断上限，不是系统真实准确率。

因此本次 3EED 复现除了报告单目标框 IoU，还应专门验证**指代目标如何从候选 query 中被选出**。建议保留 top-k 候选和每个候选的模型内评分，设计无 GT 推理时可用的选择器；GT 只在 train 监督和 val/test 评测中使用。先用完整数据验证候选召回，再定 K，不能依据 32 条 pilot 固定为 5。

## 3EED 复现模型交付给 QA 后端的最小契约

1. **输入约定**：点云格式、颜色通道、坐标系、目标描述的预处理方式、场景/目标 ID 映射。
2. **特征约定**：指定语言交互后哪一层输出；`D` 维度、每目标候选数 `K`、特征 dtype/归一化、候选顺序。最少应能输出 `[K,D]` 与相应目标指代 ID；K=1 可以，但须证明选择质量。
3. **选择约定**：训练和推理怎样用描述选候选；推理选择**不得用 GT object_id、GT box、IoU 或问答答案**。预测框与评分可写入独立诊断文件，但不得作为 LLM 的显式输入。
4. **评测与复现**：提供 checkpoint、配置、导出脚本；按完整驾驶序列隔离 train/val/test；报告实际 top1 IoU/Acc@0.25/Acc@0.5、top-k 候选召回，以及同帧双/三目标全部选对率。GT oracle 单独标注为上限。
5. **导出格式**：token bank 至少含 `scene_id, object_id, expression_id, token_path, token_index, feature_dim, checkpoint_id, selection_policy`；诊断文件可额外含预测/GT box 与 IoU。QA 样本只引用 token，不复制任何几何数值到模型输入。

当从头训练的 3EED checkpoint 替换已下载公开 checkpoint 时，QA 数据的 GT 问题与答案可以保持；需要重新导出 token 并验证 object_id 对齐。若 D、特征分布或 K 变化，projector/候选聚合器必须重新训练；Qwen LoRA 可作为初始化继续微调，但不能假设直接替换 token 后原权重仍有效。评测应分别报告 Grounding、候选选择和 QA，避免把错误混在一个总准确率里。

## 近期执行顺序

1. 完成从头训练的 3EED Grounding，并确认原仓库模型结构、loss、候选 query 与指代选择方案，先完成独立的 Grounding val 评测。
2. 双方用一小批**未用于训练的完整序列**检查 ID、描述、token 维度和 top1/top-k 目标定位，签定接口。
3. 我们替换 token bank，重训选择/聚合器与 projector，再微调/评测 LoRA；QA GT 标签继续来自离线几何标注。
4. 最后做误差拆分：GT 对象上限、Grounding 预测输入、token 打乱对照、按场景/关系类型的 QA 分数。规划任务另需未来轨迹与道路地图真值。

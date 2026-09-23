# 数据说明

## 双目标 Grounding 数据

| split | pair 数 | scene 数 | sequence 数 | 每条目标数 |
|---|---:|---:|---:|---:|
| train | 1,137 | 842 | 82 | 2 |
| val | 1,300 | 903 | 85 | 2 |
| total | 2,437 | - | - | 2 |

构建统计在 `grounding/annotations/others_linked_data/summary.json`。原始 `others` 关联边 28,823 条，能与同帧另一条真实 caption/box 精确匹配的边 4,894 条，去掉双向重复后得到 2,437 个 pair。23,929 条没有独立目标文本的 context-only `others` 未用于多目标语言监督。

## QA 版本

| 数据 | 用途 | train | val |
|---|---|---:|---:|
| `artifacts/local_qa/qa_v2` | 早期隐式 token QA | 见各目录 summary | 见各目录 summary |
| `artifacts/local_qa/qa_v3` | 多参照 QA 和后续对齐源 | 见各目录 summary | 见各目录 summary |
| `qa_correct_others_v4_contrastive` | 只保留两目标均 IoU≥0.25，测 token 能力上限 | 3,610 | 615 |
| `qa_all_others_v4_contrastive` | 不做 IoU 筛选，端到端评测 | - | 1,638 |
| `scene_reasoning_qa_v1` | 已废弃：原始 caption 进入问题，存在直接位置文本泄漏 | 2,656 | 2,862 |
| `scene_reasoning_qa_v2_noleak` | 1–5 目标、仅角色文字 + 隐式 token 的推理与局部规划 | 2,656 | 2,862 |

### 场景级 Grounding v5

| split | 帧数 | 目标总数 | 目标数分布 |
|---|---:|---:|---|
| train | 2,701 | 3,687 | 1:1858, 2:709, 3:125, 4:9 |
| val | 2,708 | 3,794 | 1:1798, 2:773, 3:106, 4:23, 5:8 |

5,518 条 v2 QA 由 3,506 条多干扰物关系题、271 条两步关系链、271 条距离排序后关系题和 1,470 条局部避障规划题组成。`qa.jsonl` 中的自然语言输入只有 A–E 角色和任务模板；opaque `object_id` 只用于把角色对齐到 token，不会由 `train_scenario_qa.py` 编码进 LLM。原始 caption、类别和 GT box 只在 `private_gt.jsonl` 中。

v4 包含三类问题：

- `relative_location`：A 相对 B 的 left/right/front/behind 组合。
- `which_is_left`：A/B 谁更靠左。
- `closer_to_ego`：A/B 谁更靠近自车。

`joint/qa.jsonl` 是公开模型输入；`private_gt.jsonl` 保存几何和 IoU 诊断，不应拼入 LLM prompt。`shuffled/` 是同类别 donor token 打乱对照，`swap_eval/` 是 A/B 交换逻辑一致性对照。

## token 文件

`artifacts/remote/others_linked_tokens/{train,val}/rank00/batch*.npz` 保留 batch 级原始导出。一个典型 batch 的主要字段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `target_count` | `[B]` | 当前均为 2 |
| `object_token` | `[B,2,288]` | soft selector token |
| `contrastive_object_token` | `[B,2,288]` | v4 使用的 token |
| `query_index` | `[B,2]` | soft query 索引 |
| `contrastive_query_index` | `[B,2]` | contrastive query 索引 |
| `pred_center`, `pred_size` | `[B,2,3]` | 评测/对齐用，不输入 LLM |
| `gt_box` | `[B,2,7]` | 评测用，不输入 LLM |
| `utterance` | `[B]` | 联合文本 |
| `object_ids_json` | `[B]` | 目标 ID 对齐 |

## 数据局限

- 当前是双目标数据，没有证明三目标以上训练效果。
- 双目标 pair 由公开 `others` 关系和两条真实单目标 caption 重建，不等同于论文未公开的原生组合式多目标问句。
- 端到端 QA 只覆盖能与 QA v3 对齐的 508 个 val pair，还没有覆盖全部 1,300 个 val pair。
- 这些 token 和标注由 3EED/Waymo 衍生，需遵守原始数据许可。

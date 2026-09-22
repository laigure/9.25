# 多物体 QA 是否必须改成联合多目标 Grounding

## 2026-09-21 更新：联合多目标成为下一阶段主线

用户决定做一次输入整句、同时输出 A/B/C 多个目标的 Grounding。此前“分别运行单目标”的方法保留为比较基线，不再作为最终模型定义。

当前新增的 `build_multi_grounding.py` 从已有 neutral bank 和 GT box 生成**自建启动数据**；它不是 3EED 官方缺失的 multi-object 标注。每句写入 2 或 3 个物体描述，记录顺序一致的 `target_spans`、`object_ids` 和 GT 框。按 QA 的 sequence 级 train/val/test 拆分，禁止同一序列跨划分。生成量为 train 908、val 614、test 397；QA test 序列不进入 Grounding 训练或调参验证。

首轮训练与测试已完成，详见 `MULTI_GROUNDING_EXPERIMENT_REPORT.md`。验证集选择的 soft-token 文字选框在独立 test 上全目标 Acc@0.25 为 5.79%，同组合逐目标单目标基线为 14.61%。因此“联合多目标”是下一阶段研究方向，**当前 checkpoint 尚不替换 QA 主链**。

修改后的 loader 按每个目标自己的文字片段建立 positive map，避免同类物体共用名词导致框与文字错配。多目标评测对 Top-1 使用一对一 query 分配；`bbf` 与 `total_acc` 要求所有目标均命中，`bbs` 继续统计逐目标命中。训练从公开单目标 checkpoint 只加载模型参数，以独立优化器微调。当前短程运行用于验证端到端训练与评测，尚不能当作论文官方多目标指标。

**后续必要工作**：导出每个角色对应的最终 decoder query token；用单目标多次调用与联合模型在同一 test 集比较；若需要真正关系语言共同指代，应补充带关系句的人工/严格标注数据，当前简单拼接描述尚不能验证这件事。

## 两种任务不要混淆

1. **当前 QA 的多物体输入**：题目有 A、B（或 C）各自的描述。分别用同一个单目标 Grounding 模型对同一帧运行 `scene + description_A → token_A`、`scene + description_B → token_B`，再把多个 token 送入 QA。`scenario_qa/qa.jsonl` 的 `object_refs` 和 `train_scenario_qa.py` 都按此设计。模型参数是同一套，运行/导出按目标描述分开；因此一道多目标 QA **不要求** Grounding 一次输出所有目标。
2. **联合多目标 Grounding**：输入一整句含多个指代的文字，模型一次输出多个目标框和各自的 query token；训练样本需有每个目标的 GT 框及对应词 span，并通过一对一匹配监督。这是更强的任务，也是 3EED 论文定义的 multi-object grounding。

## 3EED 仓库真实支持程度（2026-09-21 核对）

- 论文明确写单句多指代、每目标一个 positive map、Hungarian matching，并要求所有目标都定位正确才算成功。
- 仓库有 `scripts_multi/train_multi_3eed.sh`、`val_multi_3eed.sh`；`Joint3DDataset.waymo_multi_annos` 和 `getitem_waymo_multi` 可读取多个 GT box；`compute_hungarian_loss` 对所有有效 GT box 构造目标并匹配 query。因此不能说“架构只能训练一个目标”。
- **现成流程尚不能直接宣称可复现联合多目标**：远程数据缺 `waymo_multi_train_info.pkl` / `waymo_multi_val_info.pkl`；`GroundingEvaluator.evaluate()` 总会调用 `evaluate_bbox_by_contrast`，其中 `assert num_obj == 1`；当前 `utils/qa_token_export.py` 也只支持一目标。`val_multi_3eed.sh` 还留有空的 `--checkpoint_path`。需要先补齐标注、评测和导出接口，不能直接套现有单目标成绩。

## 对当前研究的决定

- 当前阶段先用 A/B/C 各自描述分别 Grounding，保留 QA 数据集中稳定的 `scene_id` 和每个角色的 `object_id`。分别导出 token、预测框与置信度，记录同帧**全部目标都选对**的比例和是否选到同一个候选；问答答案继续由 GT 几何离线生成。
- 已有 v3 val 里，双目标 QA 的输入目标都达到 IoU≥0.25 仅约 9.5%，所以拆成多次调用在功能上可行，但准确率是现实瓶颈。不要把“模型能输出 256 个候选框”误当成“能从一句话自动绑定 A/B 到正确两个框”。
- 若最终目标是开放式单句多指代（例如“黑车在白车左边吗”且不预先分出 A/B 描述），再投入联合多目标 Grounding：标注各指代 span 与 GT object_id，修正多目标评测为每目标 top1 和 all-target-correct，并导出每个角色的对应 query token。也可先做问题指代解析→两次单目标 Grounding 的强基线，再与联合模型比较。

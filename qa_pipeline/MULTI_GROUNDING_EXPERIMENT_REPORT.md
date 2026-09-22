# 联合多目标 Grounding 首轮实验（2026-09-21）

## 任务与输入输出

一帧 Waymo 点云加一段同时描述 2–3 个物体的文字（`Object A: ... Object B: ...`）输入一次 3EED，输出每个角色对应的独立 3D 框及最终 decoder query token（288 维）。训练用有序 GT 框和各描述的文字 span 做 Hungarian 匹配。推理选框只用模型的文字得分；离线 GT 只用于训练与评估。

本轮是**自建 neutral-reference 启动数据**，不是 3EED 论文官方多目标数据。样本描述是并列拼接，尚未覆盖关系句中的联合指代。

## 数据与过程

| split | 样本 | 场景 | 用途 |
|---|---:|---:|---|
| train | 908 | 604 | 微调 |
| val | 614 | 320 | 选 epoch 和文字分数 |
| test | 397 | 295 | 最终比较 |

三个 split 的 Waymo sequence 互不重叠。公开单目标 `ckpt_6384.pth` 只初始化模型参数，优化器重新开始；4090 上 batch 4、学习率 `2e-5`、10 epoch。以 val 全目标 Acc@0.25 选第 6 轮。训练/验证每轮约 1 分钟/2 分钟。

## 主要结果

“全目标命中”要求同一句中的每个目标都与各自 GT 框达到 IoU 阈值，Top-1 query 对不同角色强制一对一。soft-token 选择头在 val 上优于 contrastive，先固定后再评 test。

| 方法 | val 全目标 @0.25 | test 全目标 @0.25 | test 全目标 @0.5 |
|---|---:|---:|---:|
| 单目标模型分别调用（同样的目标组合） | 7.98% | **14.61%** | **5.29%** |
| 联合模型，contrastive 选框 | 2.12%* | 3.53% | 2.77% |
| 联合模型，soft-token 选框 | 6.51%* | 5.79% | 3.53% |

\* 以固定 ep6 单独重评 val；训练时 ep6 val contrastive 是 3.58%，点云抽样导致轻微波动。旧方案的对照来自 neutral bank 里同一组物体的单目标预测 IoU；它并非同一联合输入文字，但目标组合完全相同。

诊断：首次 test 评测的 contrastive 逐目标命中为 9.41%；离线使用 GT 中心，在每目标最近 16 个候选中寻找命中框，逐目标达到 35.59%、全目标达到 28.21%。最近 16 候选结果只是候选覆盖的**下界和诊断**，不是无 GT 推理分数；说明候选框和文字绑定都有改进空间。soft-token 已改善文字选框，但仍没有超过分别调用。

## 产物与限制

- 生成脚本：`qa_pipeline/build_multi_grounding.py`；三份 pkl：`qa_pipeline/artifacts/multi_grounding/`；多目标读取/评测：`3EED/src/joint_det_dataset.py`、`3EED/src/grounding_evaluator.py`。
- 远程 checkpoint：`/root/autodl-tmp/3eed_data/multi_grounding/bootstrap_run/Train_waymo-multi_Val_waymo-multi/0921_0837/ckpt_epoch_6.pth`。训练日志同目录上级 `train.log`；测试日志 `test_soft_eval.log`。
- 最终 soft-token：`/root/autodl-tmp/3eed_data/multi_grounding/tokens_test/`，100 个 NPZ 覆盖 397 样本、829 目标。每个目标保存 `object_ids_json`、`query_index`、`object_token`、`pred_center/size` 与 GT 框；GT 框只供分析。`verify_multi_export.py` 全量验证各角色的 query 唯一、特征为 288 维。早期 contrastive token 保留在 `tokens_test_contrastive/`。
- 目标 span 目前来自明确的 A/B/C 输入模板，可由输入文字解析，不需要 GT 框；开放式问题仍需角色/短语解析。拼接式样本规模小且欠缺关系语言，是训练数据的主要限制。**当前不把该 checkpoint/token 用作 QA 主链**；保留旧单目标逐个定位基线。

## 下一步

正确定位条件下的 QA 诊断已执行，详见 `CORRECT_TOKEN_QA_DIAGNOSTIC.md`：89 道同题配对测试中，联合正确 token 为 44/89、同类别打乱为 45/89，冻结 QA 模型未显示 token 收益。

1. 检查哪些组合的两个文字描述不足以区分目标，剔除含糊标注或补充更具体的描述；增加真正包含关系句的人工核对样本。
2. 用 val 比较 soft-token、contrastive 和联合分数，诊断每目标 Top-1/Top-5 与候选覆盖；仅用 val 决定下一版选择策略。
3. 在同一 test 划分上同时报告逐目标及全目标准确率，超过“分别调用”的基线后，再接入 QA 的多 token 后端。

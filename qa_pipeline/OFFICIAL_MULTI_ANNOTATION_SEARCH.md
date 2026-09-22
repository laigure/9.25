# 官方多目标标注调查报告（六步计划第 1 步）

日期：2026-09-21。结论先行：**官方 `waymo_multi_{train,val}_info.pkl` 从未公开发布，作者的公开回复是"以后再放出"**；作为替代，本步用已发布数据的原始 caption 在**官方格式**下重建了多目标标注集（train 903 / val 529 / test 386），可与现有 loader、训练脚本、评测口径直接对接。

## 1. 搜索过程与证据（都做了核验，不是猜测）

| 检查对象 | 方法 | 结果 |
|---|---|---|
| HuggingFace `RRRong/3EED` | API 列文件 | 只有 5 个文件：`3eed_dataset.zip`(6.7G)、`baseline_ckpt_pred.zip`(535M)、README、croissant、.gitattributes；无 pkl、无 multi |
| 主数据包 `3eed_dataset.zip` | HTTP Range 直接读 zip 中央目录（不下载整包） | 81,687 个条目：0 个 `.pkl`、0 个 multi 文件；只有逐帧 `image.jpg / lidar.npy / meta_info.json` |
| `baseline_ckpt_pred.zip` | 同上 | 36,719 个条目，仅单目标 `final_6384` 运行（即我们用的 `ckpt_6384.pth`）；6 个 "multi" 命中全是作者代码备份里的 `scripts/train_multi_3eed.sh` |
| GitHub `worldbench/3EED` | 本地 clone 与 `git ls-remote` 比对 | 远端 HEAD 与本地一致（843a901，2025-12-26）；仓库无 multi 数据文件，只有 `scripts_multi/train_multi_3eed.sh`、`val_multi_3eed.sh`（后者 `--checkpoint_path` 为空）与 `data/splits/multiwaymo_{train,val}.txt` |
| GitHub Issue #2 "Waymo-Multi Data" | API 读全文 | 2025-11-04 用户问 pkl 是否缺失/会否发布；作者 iris0329 回复 **"The Waymo-Multi part will be organized and released later—we've been quite busy recently."** 至今（2026-09-21）HF 数据集最后更新仍是 2025-10-25 |
| 其它 HF 仓库 | 搜 "3eed" | 只有 `yuzaiyangy/3EED`，文件名与大小和官方的完全一致（镜像），无新内容 |
| 全网 | 搜特征文件名 `waymo_multi_*_info.pkl` | 无任何第三方镜像或转载 |

**所以：复现论文 Table 3 的前置条件（拿到官方多目标标注）目前不成立。** 可选的补救是给作者提 issue / 发邮件索取；在拿到之前，本阶段只能用重建集推进，且必须明确标注"非官方标注、数字不可与 Table 3 直接等同"。

## 2. 一句话 ↔ 多个目标框：官方格式（按仓库 loader 代码逐行核对）

- 文件：`data/waymo_multi_{split}_info.pkl`，`pickle.load` 得到 `list`；每项是一句 caption 配 N 个目标框。
- 每项字段：
  - `caption`（str）：一句话，可同时指代多个物体；
  - `segment_name`、`frame_name`（如 `"0034_0"`，loader 会补零到 4 位）；
  - `bbox3d_obj_1..N`：7 维框 `[x, y, z, dx, dy, dz, heading]`；
  - `bbox2d_obj_1..N`（可缺，缺失时该目标不填 2D 信息）；
  - `class_obj_1..N`：类名，取值 car / pedestrian / bus / othervehicle / truck / cyclist；
  - `target_spans`（可选，强烈建议）：`[(begin, end), ...]` 字符级区间，**必须与框一一对应**，用来给每个目标建 token 级掩码；
  - `object_ids`（可选）：每目标的稳定 ID。
- loader 读取后生成：`positive_map [N, 256]`（逐目标二值化并归一化）、`box_label_mask`、`gt_bboxes [132, 7]`、点级 `point_instance_label`。校验规则：每个目标必须有非空 span，否则直接报错。
- 训练：逐目标做 Hungarian 匹配，损失按目标平均（论文 B.2）。评测：**所有目标都命中才算该句正确**（joint correctness），报 Acc@25 / Acc@50 / mIoU，按类别分列（Car / Pedestrian，Vehicle 平台）。
- 官方训练口径（论文 B.3 与 `scripts_multi/train_multi_3eed.sh` 一致）：**从零训练**（不加载单目标权重）；全部模块 lr 1e-4；**200 epoch**；batch 12；单卡 4090；lr 衰减 75/80 轮。
- 推理选框（论文 B.2）：对每个目标，用"候选框 ↔ 该目标的文字 span"的语义相似度选 Top 框；本仓库 `utils/qa_token_export.py` 用 soft-token 分数 + 匈牙利分配实现，**不依赖 GT 框**（部署需一个文字 span 解析器）。
- 官方标注怎么生成的（论文 Table 9）：把两张标注帧 + 两个类名 + 预设关系喂给 Qwen2-VL-72B，输出 `Object A: … / Object B: … / Relationship: …` 三段式描述，再人工校验。

## 3. 官方切分文件的一个问题

`data/splits/multiwaymo_train.txt` 与 `multiwaymo_val.txt` **两个文件完全相同**（MD5 均为 `a075ee542c4fb2ff6928760bbe9df015`，128 行）。也就是说官方仓库里多目标切分是同一份拷贝，真实的 train/val 划分应该在**未发布**的 pkl 内部。这批 segment 与单目标切分是同一批场景的重新划分；128 个里 126 个的点云在已发布数据中（缺 `11616035176233595745_...` 与 `11901761444769610243_...` 两个）。

## 4. 本步交付：官方格式重建集（非官方标注）

- 生成脚本：`qa_pipeline/build_official_format_multi.py`；产物：`qa_pipeline/artifacts/multi_grounding_official_fmt/waymo_multi_{train,val,test}_info.pkl` + `summary.json`。
- 数据来源：已发布 Waymo 帧的**原始 caption**（Qwen2-VL-72B 生成、经过人工校验的单目标标注）+ 同帧多个物体的 GT 框（`#gN` 编号与 QA 线一致）。
- 句子结构：`Object A: <A 的原始 caption> Object B: <B 的原始 caption>`（三目标再加 `Object C: …`）；`target_spans` 是字符级精确区间，逐个用"格式化后的 caption 子串 == 原文"断言过。
- 规模：**train 903**（807 双目标 + 96 三目标，609 场景，79 序列）、**val 529**（468+61，324 场景，39 序列）、**test 386**（357+29，295 场景，40 序列）。
- 切分：沿用 QA v3 的序列级划分，train/val/test **序列交集全为 0**（QA test 序列不会进入 Grounding 训练与验证）。
- 本地已校验：字段完整性、7 维框、span 区间合法且与文本一致、切分隔离。**tokenizer 级校验**（每个 span 能建出非空 positive map，即 loader 的 `get_positive_map` 断言）留在远端 smoke 跑，本地没有 torch/transformers。
- 与官方多目标的差异（务必记住）：
  1. 句子是两条单目标 caption 拼接，**没有官方 `Relationship: …` 联合关系句**；
  2. 样本选取是我们定的（每场景最多 4 对 + 1 个三目标组合），官方选取规则未知；
  3. 因此它是"同格式、同场景、同评测代码"的替代品，**不能声称复现 Table 3**。

## 5. 对后续步骤的影响

1. 第 2 步小数据训练、第 3 步完整训练都在重建集上进行；训练脚本按官方 `train_multi_3eed.sh` 口径（从零、lr 1e-4、batch 12；200 epoch 对 900 条数据是否合适，先用小数据观察 loss 再定）。
2. 第 3 步指标：现有 evaluator 输出逐目标 IoU 与"全目标命中"；需补 mIoU，并按 Car / Pedestrian 分列展示，与 Table 3 同格式对照——但注明数据不同，数字不可直接等价。
3. 第 4 步导出接口已具备（288 维 token、query 编号、预测框、匹配分），只差把"置信度"字段补齐。
4. 第 5 步注意：QA 问题里的物体描述是"中性短语"，而重建集训练用的是一整段 caption；到时统一输入风格（用 QA 的角色描述拼 "Object A: … Object B: …" 提问，或训练时混入中性短语版本，脚本已留 `--caption-field neutral_query` 开关）。
5. 如果之后作者发布官方 pkl：把同一批评测脚本指向官方文件重跑即可，流程不用改。

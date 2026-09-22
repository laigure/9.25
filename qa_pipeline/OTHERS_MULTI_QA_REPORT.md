# 官方 others-multi 多目标 Grounding → 空间 QA 全链报告（2026-09-22）

本报告覆盖第四、五步：按官方 `scripts_multi` 在 `waymo-others-multi` 上完整训练 200 轮 → 导出多目标 token → 构建"定位全对"QA 库 → 训练 projector+LoRA → val 评测（joint vs shuffled）→ token 线性探针诊断。

## 1. 训练（官方脚本口径）

- 脚本：`3EED/scripts_multi/train_others_multi.sh`（单卡 4090、batch 12、200 epochs、lr 1e-4、soft-token loss + contrastive align + self-attend + augment_det）。
- 产物目录：`/root/autodl-tmp/3eed_data/multi_grounding/others_linked_run/Train_waymo-others-multi_Val_waymo-others-multi/0921_2210/`；`ckpt_epoch_200.pth`（02:19）、`ckpt_epoch_last.pth`（02:23）。
- 最终评测（epoch 195/200）：

| 指标 | 数值 | 口径说明 |
|---|---:|---|
| Eval/acc@0.25（逐目标平均） | 59.5% | 新口径 PerTargetContrast，与历史不可比 |
| JointAll-bbf@0.25 | 40.6% | 旧联合口径，可比 |
| JointAll-bbf@0.5 | 18.5% | 旧联合口径，可比 |
| mIoU | 34.8% | 可比 |

历史对照（旧联合口径同量级）：coordinate_fix ep75 13.6% / mIoU 12.2%；官方旧 run ep125 11.5% / 12.0%。本轮 40.6% / 34.8% 是数量级提升，但仍远低于单目标 80.67% 口径（不同任务、不可直接比）。

## 2. 多目标 token 导出

- 脚本：`3EED/scripts_multi/export_others_multi_tokens.sh CHECKPOINT {train|val}`；`--eval --export_qa_tokens`。
- 结果：train 285 文件 / 1137 样本；val 325 文件 / 1300 样本；每样本 target_count=2。
- token 来源与选择：最终 decoder 层 query 特征（`feature_name=last_query_features`，288 维）；每目标一个 token，选择只用 soft-token 文字 span 分数 + 唯一分配，**全程不依赖 GT 框**；npz 中同时保存 `query_index`、`match_score`、`pred_center/size`；GT 框仅供离线 IoU 过滤。

## 3. QA 库构建（只用"定位全对"样本）

- `build_correct_multi_qa.py --iou 0.25 --splits train,val`，输出 `qa_correct_others/`。
- 2437 组多目标样本里 1618 组（66%）全目标 IoU≥0.25（train 样本内 ≈97%，val 跨场景 ≈38%）。
- 配对后题量：train 2447 / val 620（每样本一题）；题型分布：relative_location 1598、which_is_left 841、closer_to_ego 628。
- 打乱对照（donor token 同 split/同类别、题目答案不变）全部 id 不同。
- QA 答案由 GT 几何生成（qa_v3 私有 GT）；LLM 输入只有问题文字 + 有序 soft token，无坐标。

## 4. LoRA 训练与评测

- `train_scenario_qa.py --mode lora`，3 epochs（306 step/epoch，共 918），loss 2.52→~0.06，val_loss 0.0893/0.0874/0.0818 逐轮下降；`best.pt` 36 MB（projector + lora 都在）。
- val（n=620，跨场景，均为定位全对样本）：

| 输入 | 总准确率 | relative_location | which_is_left | closer_to_ego |
|---|---:|---:|---:|---:|
| joint（真 token） | **0.6226** | 0.500 | 0.561 | **1.000** |
| shuffled（打乱 token） | 0.3484 | 0.204 | 0.585 | 0.409 |

打乱后总准确率掉 27 个点，说明 token 被真实使用了（不是靠文字先验蒙）。**但分题型看：距离题（closer_to_ego）100% 且打乱后掉到 41%；方向题（relative_location/which_is_left）≈随机水平，且 which_is_left 打乱后不降反升。**

补充 1：用同一个 best.pt 在 **train split（样本内）** 评测（`runs/others_lora_train_eval/eval_train.json`）：总准确率 0.7503（n=2447），relative_location 0.663、which_is_left 0.731、closer_to_ego 1.000。先验/多数类基线（同一批题）：relative_location 0.189、which_is_left 0.516、closer_to_ego 0.518（binary 题都是 ~52%）。相对位置题 66% >> 18% 多数类，说明模型确实在读 token。

补充 2：**10 轮对照训练（`runs/others_lora_e10`，其余配置与 3 轮完全相同）**。val_loss 逐轮：0.0875 / 0.0960 / 0.0849 / **0.0668（第 4 轮触底）** / 0.0727 / 0.0765 / 0.0957 / 0.1057 / 0.1061 / 0.1067；train loss 到第 9–10 轮降到 0.0003（训练集被背下来了）。best.pt 落在第 4 轮，val（n=620）：**总准确率 0.7242（+10.2pp）**，relative_location **0.654**（+15.4pp）、which_is_left **0.640**（+7.9pp）、closer_to_ego 1.000。

于是"欠训练 vs 结构问题"的答案是**两者都有**：3 轮版确实没跑够（val_loss 还在降就被截断，多训到第 4 轮白赚 10 个点）；但第 5 轮起就过拟合（train 背到 0.0003、val_loss 一路涨到 0.107），说明这套 2447 题的训练量支持不了更大的拟合，方法层面要解决的是**泛化**。方向题残差仍在：adapter 最好的方向题成绩（which_is_left 0.640）对比线性探针同口径 0.957，还差约 30 个点。

## 5. 方向题弱项诊断（本轮重点）

先排除两个嫌疑（均被数据否定）：

1. **标签和 token 坐标系不一致？** 否。全量逐视角核对 1240/1240 一致，中心最大差 0.000078 m；val pkl 全部 `coordinate_frame=source_lidar`，抽查 0024_4 旋转后与导出值完全吻合（F 484 / FL 210 / FR 346 / SL 142 / SR 58 各视角都有覆盖）。
2. **按预测框几何重判答案会不会变好？** 否。用模型自己的 `pred_center` 重算标签再评分：relative_location 0.500→0.516，which_is_left 0.561 不变，closer_to_ego 1.000 不变。方向弱项不是"标签用 GT、模型看预测"造成的。

再定位信息在不在 token 里——**线性探针（Ridge，288 维 token → 目标 (x,y)）**，train 拟合、train 内部切分选 alpha、val 一次评估，CPU 约 10 秒：

| 子集 | R² x(前后) | R² y(左右) | 备注 |
|---|---:|---:|---|
| train 样本内 | 0.939 | 0.890 | MAE 1.64 / 1.25 m |
| val 全量 | 0.585 | 0.057 | 混入 41.5% 选错目标 |
| **val 选对目标（IoU≥0.25）** | **0.900** | **0.792** | MAE 1.99 / 1.64 m（n=1520） |
| val 选错目标 | 0.089 | −0.786 | 选错时 token 指向的是另一个物体 |

同口径答题能力（val 选对目标子集）：

| 探针能力 | 数值 | 对应 LoRA 表现 |
|---|---:|---:|
| 左右命名 bin（|y|≥2.0） | 85.4%（n=1013） | relative_location 横向 ≈64% |
| 前后命名 bin | 100% | longitudinal ≈82.5% |
| 两目标左右排序（|Δy|≥2.0） | **95.7%**（n=465） | which_is_left 56.1% |
| 哪个更近 | 97.3% | closer_to_ego 100% |

**结论：288 维 token 里方向信息是线性可读、可跨场景泛化的；当前 projector+LoRA（3 epochs）只学会了"远近"，没学会"左右"。瓶颈在 adapter 训练/结构，不在 token 导出，也不在标签。**

## 6. 产物与复现

- 一键链：`qa_pipeline/posttrain_others_step5.sh`（md5 9fcaba79e3c70efb72c44ef164adfa6c）；导出脚本 md5 5041ea63af2cd6e5f01864beff427acd。
- 远端：`$DATA/others_linked_tokens/{train,val}`、`$DATA/qa_correct_others/{joint,shuffled}`、`$DATA/runs/others_lora[/_shuffled]`、`$DATA/others_linked_posttrain/{status.log,state,final_summary.txt}`。
- 探针：`qa_pipeline/linear_probe_token_xy.py`（md5 见远端 log `others_linked_posttrain/linear_probe.log`）。

## 7. 下一步（按优先级）

1. **方向题残差属于泛化问题，方案待用户确认**（三条候选，可叠加）：
   - (a) 正则化 / 容量控制：3 轮太短、5 轮起过拟合，说明要在 4 轮附近配合更强正则（LoRA dropout、weight decay、更小 rank）或只训 projector；
   - (b) 几何预训练 projector：先用线性探针同款的 token→(x,y) 回归预训练 projector，再对齐 LLM，把"方向可线性读出"这一已证事实变成归纳偏置；
   - (c) 加数据：目前的 2447 题来自 1137 个 train 样本，扩大样本量是最直接但最慢的路。
2. **阶段 E（路线规划）数据侦察已完成**：waymo 每帧 `meta_info.json` 带 `pose`（4×4）与 `timestamp`，且同一 segment 下有多个连续帧（样例 segment 有 12 帧）→ ego 真实轨迹、速度变化可以直接从连续帧 pose 推出来，**"直行/减速/停车/左绕/右绕"和"危险目标"都可以用真实驾驶轨迹+几何规则生成弱标签**，不需要额外的轨迹数据集；others-multi 的 pkl 本身只有单帧两目标+context（无速度/轨迹字段），路线规划要用原始 `3eed_dataset` 的连续帧。
3. 与论文 Table 3 的 others-multi 官方数值对齐核实（B6）。

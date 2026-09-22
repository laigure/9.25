# 3EED Waymo Grounding 复现记录

## 目标

用官方 3EED 代码从随机初始化训练 Waymo/Vehicle Grounding，在官方 val 划分评测，并核查候选 query、预测框与可用于 QA 的 token。与已下载的公开 checkpoint 分开记录。

## 环境与适配

- 远程主机：AutoDL 容器；单张 RTX 4090，24 GB；实际 `agiclass` 环境为 PyTorch 2.1.0+cu121、Transformers 4.44.2；代码 `/root/3eedqa/3EED`；数据 `data/3eed` 链到完整 3EED 数据集。
- 仓库 `scripts/train_waymo.sh` 写死 `CUDA_VISIBLE_DEVICES=4`、batch 24。此机只有 GPU 0；复现脚本改为单卡 batch 4，保留 6 层 decoder、RGB、官方 loss 选项、学习率和 100 epochs。val 改为每 5 epochs，checkpoint 每 10 epochs（每份约 705 MB；磁盘仅剩约 14 GB）；这些是**与官方脚本不同的配置**。
- 官方已下载 checkpoint `/root/autodl-tmp/3eed_data/checkpoints/ckpt_6384.pth` 只作独立参考评测；从头训练不得加载它。
- 之前加入的 `return_query_features` 只在显式请求时保存最终 decoder query；默认训练 forward 行为不变。既有仓库 diff 还包括 QA 导出工具和 Windows/Linux 行尾差异；运行前需核对训练代码不受影响。

## 操作进度

1. 2026-09-20：确认远程 SSH 可用、GPU 空闲、数据和 checkpoint 在位；本地核对 `train_dist_mod.py`、`main_utils.py`、官方 Waymo 训练脚本和模型 decoder。建立 `run_3eed_waymo_repro.sh`。训练尚未启动。
2. 2026-09-20：检查远程仓库 diff（忽略行尾差异）后确认已有修改只扩展 `--eval` token 导出，默认训练 forward 未变。同步脚本至 `/root/autodl-tmp/3eed_data/run_3eed_waymo_repro.sh`。
3. 2026-09-20：启动独立的一轮 `--debug` 短训练，batch 4、保存/验证每轮，PID 37701；日志 `/root/autodl-tmp/3eed_data/grounding_repro/smoke.log`，输出目录 `grounding_repro/Train_waymo_Val_waymo/smoke_b4_seed0/debug/`。该 smoke 使用重复样本，只验证训练链路，不作为复现指标。
4. 2026-09-20：第一次 smoke 在模型导入阶段失败，base Python 缺 `transformers`，**没有进行任何训练**。确认先前评测用的 `agiclass` 环境包含 PyTorch 2.1.0+cu121 与 Transformers 4.44.2；已修正复现脚本 PATH，准备重试。
5. 2026-09-20：用 `agiclass` 重跑 smoke 成功：随机初始化、11 个训练 batch 完成前向/反向，保存 705 MB checkpoint，随后执行 val；仅重复样本的 Acc@0.25/0.5 均为 0，属预期，不能当模型性能。峰值观察 GPU 内存约 6.4 GB，无 OOM/NaN。正式脚本 checkpoint 频率改为每 10 epochs，以免磁盘填满。
6. 2026-09-20：**从头训练已启动**，外层 PID 38466，`torchrun` PID 38467，训练进程 PID 38533；日志 `/root/autodl-tmp/3eed_data/grounding_repro/full.log`，输出目录 `grounding_repro/Train_waymo_Val_waymo/full_b4_seed0/`。完整训练 100 epochs，Waymo train 约 3,682 条描述、batch 4，每轮 920 step；此时正在 epoch 1，最终结果尚不可报告。
7. 2026-09-20：独立核对**公开 checkpoint**先前完整 Waymo val（3,782 条）的官方评测日志：Acc@0.25=0.8067、Acc@0.5=0.4989、mIoU=0.4674。这是已下载 checkpoint + 原始完整表达的结果，不是当前从头训练的成绩，也不同于中性化描述 QA 的 47.5% Acc@0.25。
8. 2026-09-20：抽查官方评测导出的首个 8 样本 `.npz`：`object_token [8,288]`、`pred_center [8,3]`、`pred_size [8,3]`、`query_index [8]`，文件标记 `feature_name=last_query_features`。模型内全部是 256 个候选 query；导出器按语言投影评分选同一索引的特征和预测框。**注意该官方导出选 query 使用标注的正文本 span (`positive_map`)**；这是有标签评测/诊断设置，不能直接声称部署时无需 span 标注。
9. 2026-09-20：正式训练第 1 轮完成，920 step、耗时 226.80 秒，当前已进入第 2 轮；训练 loss 有限、GPU 占用约 6.3 GB，未见 NaN/OOM。按当前速度 100 轮加每 5 轮验证需数小时；目前不能宣称复现结束。
10. 2026-09-20：新增并同步 `qa_pipeline/run_3eed_posttrain.sh`，监视训练外层 PID 38466 的进程 PID 39408，日志 `/root/autodl-tmp/3eed_data/grounding_repro/posttrain.log`。训练正常生成 `ckpt_epoch_last.pth` 后，自动在完整 Waymo val 上重新评测并导出最终 token，再生成 `shape_report.json`；若无最终 checkpoint 则报错退出。`--eval` 导出仍使用标注文本 span，仅作 Grounding 基线审计。
11. 2026-09-21 00:41:43（北京时间）：按新任务指令**温和停止**本从头训练及监视任务，产物全部保留、未清理。停止前训练完成至第 30 轮（`ckpt_epoch_30.pth` 于 00:35 保存）；第 30 轮验证进行到 63%（600/946）时中断；最后完成的验证为 epoch 25（00:16:23）：Acc@0.25 0.7700 / Acc@0.5 0.4326 / mIoU 0.4263。停止方式：torchrun(38467) SIGINT → worker(38533)、外层(38466) 退出；监视(39408) SIGTERM；核对无残留进程，GPU 空闲。`ckpt_epoch_last.pth` 与 `final_waymo_tokens/` 均未产生。**本方向（从头复现）到此为止**，后续改用公开 checkpoint 冻结后做隐式空间 QA 对照实验（见 `WORK_LOG.md` 第 35 条）。

## 验收

1. 短训练能完成 forward、backward、checkpoint 和 val，确认无 OOM/NaN。
2. 正式训练给出独立 checkpoint、准确记录训练配置和 val Grounding Acc@0.25/0.5；报告与官方 checkpoint 的差异。
3. 同一批 val 样本检查 `[B,256,288]` 最终 query、`[B,256,3]` 中心/尺寸、语言候选评分的索引一一对应；实际选中目标质量与 top-k/oracle 上限分开报告。

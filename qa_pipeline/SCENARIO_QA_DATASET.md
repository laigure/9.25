# 3EED 隐式 token 空间问答数据集 v2

## 目标与数据流

公开 3EED Grounding 模型先接受点云和去空间提示的目标描述，输出最后一层 288 维 query token。问答模型只接收自然语言问题与 1–3 个经共享 projector 映射的目标 token。GT 3D box 只在离线制标和评测时使用，**不输入问答模型**。模型训练为 Qwen2.5-7B-Instruct + projector + LoRA。

```
点云 + 中性目标描述 → 冻结的 3EED Grounding → 288D object token
                                        ↓
问题文本 + 1–3 个 token → 共享 projector → Qwen LoRA → 完整句子

GT box → 离线几何规则 → 标准答案/私有评测标签
```

## 目录

- `qa_v2/neutral_bank.jsonl`：目标级导出审计，包括 GT/pred box 和 IoU；训练问答时不读取。
- `qa_v2/tokens/waymo_tokens.npz`：6,130 个 288 维 query token 和对应 object_id。
- `qa_v2/scenario_qa/qa.jsonl`：训练/验证/测试的公开问答记录。
- `qa_v2/scenario_qa/private_gt.jsonl`：与 qa_id 对齐的 GT box 与答案类别，仅供离线评测。
- `qa_v2/scenario_qa/summary.json`：数量、序列数与输入规则。
- `qa_v2/runs/full_lora/`：正式训练日志、checkpoint、val/test 结果。

上述远程目录前缀为 `/root/autodl-tmp/3eed_data/`。构建/训练脚本位于本地 `qa_pipeline/` 并同步到远程 `/root/3eedqa/qa_pipeline/`。

## 题型

| 场景 | 问题示意 | 输出示意 | train / val / test |
|---|---|---|---:|
| 自车与目标 | `Object A: a red sedan. Where is Object A relative to the ego vehicle?` | `Object A is to the left and in front of the ego vehicle.` | 2629 / 1463 / 1307 |
| 两目标相对位置 | `Object A: a white car. Object B: a black sedan. Where is Object A relative to Object B?` | `Object A is behind Object B.` | 655 / 462 / 304 |
| 三目标比较 | `Object A: a truck. Object B: a car. Object C: a cyclist. Which object is closer to Object C, Object A or Object B?` | `Object B is closer to Object C.` | 80 / 55 / 29 |

合计 6,984 条。官方 Waymo train 的 94 个序列用于 QA train；官方 Waymo val 的 93 个序列按固定种子分为 QA val 46 个、test 47 个。三个集合没有共同驾驶序列。含糊的方位边界和距离差距过小的比较题不制标；不根据 Grounding 预测 IoU 挑选测试题。坐标方向采用 Waymo 的 +x 前、+y 左、+z 上约定，见 [Waymo 官方数据格式](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto)。

## 当前局限

- 中性描述降低了定位准确率：全量 train/val 的 IoU≥0.25 分别为 0.5728/0.4750。问答分数同时受 Grounding 误差影响，应报告定位质量与各题型问答准确率。
- 三目标题仅 164 条，总分容易被单目标题主导；应按题型报告，并与训练集多数类基线比较。验证/测试多数类基线总分为 0.3530/0.4195。
- 本版不生成路线规划题。3EED 当前转换数据不提供未来自车轨迹、道路可行驶区域或动作真值；仅凭单帧物体框不能可靠生成建议路线或避障决策标签。规划版需要补充时序位姿、动态目标轨迹与地图/道路监督。Waymo 另有带未来轨迹的 [End-to-End Driving 数据格式](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto)，属于后续扩展数据源。

## 复现命令

```bash
python /root/3eedqa/qa_pipeline/build_scenario_qa.py \
  --bank /root/autodl-tmp/3eed_data/qa_v2/neutral_bank.jsonl \
  --out-dir /root/autodl-tmp/3eed_data/qa_v2/scenario_qa

python /root/3eedqa/qa_pipeline/train_scenario_qa.py \
  --out-dir /root/autodl-tmp/3eed_data/qa_v2/runs/full_lora \
  --epochs 3 --batch-size 2 --grad-accum 4 --eval-splits val,test
```

持续进度记录见 `WORK_LOG.md`。

## 当前实测（最佳第 2 轮 checkpoint）

完整验证集总体准确率 0.3601；ego 0.3978、relative 0.2316、closer 0.4364。完整测试集总体准确率 0.4201；ego 0.4422、relative 0.3322、closer 0.3448。训练集多数类的 val/test 总基线为 0.3530/0.4195。当前模型几乎没有超过基线，**不能说明已学会可靠空间推理**。输出严重偏向固定答案。独立的线性 token 探针在 ego 验证集达到 0.5284，说明 token 中有一部分位置线索，而当前 LoRA/projector 未充分提取。355 条验证探针中，正常 token 准确率 0.3408，按类别打乱 token 后为 0.3324；只有 19 条预测发生变化，进一步表明当前后端几乎没有利用 Grounding 特征。

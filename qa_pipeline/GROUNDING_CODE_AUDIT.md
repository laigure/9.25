# 3EED 单目标与多目标 Grounding 代码核查

日期：2026-09-21

## 结论

师兄对“公开代码不能直接作为完整多目标 Grounding 使用”的判断是正确的。

但更准确的说法是：

- 公开默认数据通路和已发布 checkpoint 是单目标 Grounding。
- 仓库中有 `waymo-multi` loader、训练脚本和支持多个 GT 的 Hungarian loss 骨架。
- 公开的多目标闭环没有完成：正式标注未发布，原评测器断言 `num_obj == 1`，验证脚本没有 checkpoint，multi loader 还有路径和文字对齐问题。
- 因此，我们现在做的是在 3EED 主干上补全多目标数据、监督、评测和 token 导出，不能称为原仓库开箱即用的功能。

## “把后面的 token 也 positive”应该怎样实现

不能把多个目标词全部写进同一行 positive map。正确监督结构是：

```text
caption: Object A: black car. Object B: white car.

positive_map[0] = Object A 对应的文字 token
gt_bboxes[0]    = 黑车框

positive_map[1] = Object B 对应的文字 token
gt_bboxes[1]    = 白车框

box_label_mask[:2] = 1
```

训练时 Hungarian matcher 把两个不同 decoder query 分别匹配到两个 GT 框，并使用各自的 positive-map 行计算语言分类、框回归和对比损失。

错误做法是：

```text
positive_map[0] = Object A token + Object B token
gt_bboxes[0]    = 只有一个框
```

这样模型学到的是“多个词共同指向一个框”，仍然不是多目标 Grounding。两辆同类别汽车尤其需要 A/B 的精确 span，不能只按单词 `car` 合并。

## 官方代码实际情况

### 已经具备的多目标骨架

1. `Joint3DDataset` 注册了 `waymo-multi`。
2. `waymo_multi_annos` 能读取 `bbox3d_obj_1..N`。
3. `box_label_mask` 能标记 N 个有效 GT 框。
4. `compute_hungarian_loss` 根据 mask 构造 N 个 target。
5. `HungarianMatcher` 支持 `num_target_boxes > 1`，能做一对一 query 分配。

### 原仓库没有完成的部分

1. `waymo_multi_{train,val}_info.pkl` 没有公开。
2. 默认 `load_3eed_annos` 仍将每条 `ground_info` 展开成一个表达和一个目标框。
3. 原 `evaluate_bbox_by_contrast` 明确断言 `num_obj == 1`，多目标验证会中止。
4. 原 `val_multi_3eed.sh` 的 `--checkpoint_path` 为空。
5. 原 multi loader 会原地重复替换路径，第二轮可能得到 `data/3eed/3eed/...`。
6. 原 loader 根据类别词出现位置自动生成 positive map，不能稳定保证第 N 个文字 span 与第 N 个框一一对应。

## 本项目已经补的部分

1. 每个目标保存独立 `target_spans`，并逐行对应 `bbox3d_obj_N`。
2. loader 强制检查 span 数量、框数量及非空 positive map。
3. 修复 multi 路径重复替换。
4. 增加多目标唯一 query 分配、联合 Acc、逐目标 Acc/mIoU。
5. 导出每个目标独立的 288 维最后层 decoder token、query index、预测框和匹配分。

## 本次审计新发现并已修复的问题

### 1. 五视角坐标没有统一

复建数据包含 LiDAR 视角 0–4；原 multi loader 没有执行单目标 Waymo loader 使用的前视旋转。虽然点云和框仍在同一原始坐标系，几何监督不会完全错位，但 `left/right/front` 等相机相对语言在不同视角下不一致。

修复方式：复建记录明确写入 `coordinate_frame="source_lidar"`；loader 只对带该标记的数据同时旋转点云和全部有效框。对未来可能发布、坐标定义未知的官方 pkl 不做自动旋转，避免重复变换。

验证：train/val/test 各抽取非前视样本，重算所有 GT 框并以 `1e-6` 容差比较，全部通过。

### 2. 总 mIoU 混入所有 decoder 层

旧扩展评测会把 proposal 和各 decoder head 的 IoU 一起累加。修复后总 mIoU 和预测记录只使用 `last_` 最终 decoder 层。

### 3. 重复类别被错误折叠

旧统计用集合将 `car+car` 折叠成 `car`。现已保留重复类别，联合结果会正确显示为 `car+car`。

### 4. 所有目标点都被标成实例 0

上游 `get_waymo_multi_target_boxes` 虽然循环读取了多个目标框，却执行：

```python
point_instance_label[fg_mask] = 0
```

这会把第二、第三目标框内的点也标成目标 0。`compute_points_obj_cls_loss_hard_topk` 会使用该编号为每个 GT 选择 query seeds，因此旧代码的 query-point 辅助损失实际上只正确监督第一个目标。这与师兄指出的“代码只监督第一个目标”一致。

现已改为：

```python
point_instance_label[fg_mask] = i
```

真实数据探针已观察到实例编号 0 和 1；合成 loss 测试确认两个文字 span、两个框会匹配两个不同 query，实例 0/1 均可进入 query-point loss 并正常反向传播。

抽样还显示并非每个 GT 框都有内部点：train/val/test 前 128 条的目标覆盖率分别约 47.5%/22.1%/16.9%。进一步同时检查原始完整点云和采样后的 16K 点，覆盖率完全相同，`visible_targets_lost_by_sampling=0`。因此问题不是随机采样丢掉稀疏目标，而是这些框在对应原始点云内本来就没有点。没有框内点的目标仍参与语言、框回归和 Hungarian loss，但无法获得 query-point 辅助监督。该项属于复建数据的可见性限制，需要在完整结果中单独分析，不能再误判为实例编号错误。

### 5. 单进程 loss 测试会错误调用分布式 world size

`SetCriterion` 原来仅在 all-reduce 前检查分布式是否初始化，随后仍无条件调用 `dist.get_world_size()`。正式 DDP 训练不受影响，但普通单进程测试会崩。现已在非分布式环境使用 world size 1。

### 6. 原复建文字 span 混入了参照物

早期复建记录把每个目标的整段 source caption 都标为 positive；一段描述常会同时提到目标、参照车辆和其他物体，导致同一行监督不再只代表对应 GT 框。现保留完整描述作为模型上下文，但每个 `positive_map[i]` 只覆盖该目标的 `neutral_query`；无法精确定位短语时才退化到该段内的目标名词。重建后 train/val/test 的 span 均非空且互不重叠。

### 7. 多目标推理必须做唯一 query 分配

分别对每个目标独立 `argmax` 可能把同一个 query 同时交给两辆车。评测和 token 导出现在都对“目标 × query”得分矩阵执行 Hungarian 一对一分配；每个目标的 span 权重归一化，避免较长短语只因 token 数量更多而占优。导出的 288 维 token、预测中心和尺寸使用同一个最终 decoder query 索引。

### 8. 验证集随机点采样导致指标不可复现

训练集继续随机采样；val/test 改为根据 `split:scan_id` 的 SHA1 生成固定采样种子。相同样本重复读取完全一致。当前 20:41 已启动的进程在导入该修复前启动，因此它的周期性验证仍有轻微采样波动；训练权重不受影响，最终会用更新后的代码单独做确定性 val/test 评测。

### 9. 两个命令行开关实际上没有生效

当前 3EED 数据函数最终只返回 xyz，模型主干也固定按 3 通道建立，因此脚本中的 `--use_color` 没有把反射率或 RGB 输入模型。`--augment_det` 在 `Joint3DDataset` 中只保存为成员，multi 路径没有调用相应增强。这不会造成张量错位，但必须在实验说明中写清，不能把本次训练描述成使用了颜色或检测增强。

### 10. 两个评测 head 曾使用不同分母

扩展评测最初把 soft-token (`bbs`) 的 Acc 按目标数统计，却把 contrastive (`bbf`) 的 Acc 按“整句话是否全部正确”统计，导致两个 head 的同名 Acc 不可比较，训练日志中的总 Acc 也偏低。现已统一：普通 Acc 对两种 head 都按每个目标统计；`JointAll-bbs` 和 `JointAll-bbf` 另行报告一条表达中的所有目标是否同时正确。该修复只改变评测计数，不影响损失或 checkpoint。

## 旧实验如何处理

- 公开单目标 checkpoint 的 Waymo 单目标结果不受影响，仍可作为单目标基线。
- 自制拼接数据的 5.79% 继续只作为早期原型记录。
- 坐标修复前复建集训练得到的 11.53% 及 epoch 150 checkpoint 标记为诊断版本，不再作为正式结果，也不接 QA。
- 坐标定义改变后必须从零训练，不能从旧 checkpoint 续跑。

## 修正版本运行状态

- 数据：train 903 / val 529 / test 386，每条 2–3 个目标。
- multi loader 三个 split 均通过真实点云 smoke。
- 20 epoch debug 训练完成；loss 明显下降，侧视框旋转测试通过。
- 19:15 启动的坐标修正版训练在二次审计发现实例编号错误后停止并标记作废。
- 点实例编号修复后的 20 epoch smoke 已完成：loss 正常下降，联合 Acc@0.25 曾达到 2.27%，用于证明链路可训练，不作为正式指标。
- 最终修正版必须从零训练，输出目录使用 `instance_fix_run`，避免与旧实验混淆。

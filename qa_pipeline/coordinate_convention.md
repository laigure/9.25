# 3EED 坐标系约定（QA 标签使用的定案）

本文回答生成 left/right/front/behind 标签前必须确认的问题：x/y/z 正方向、ego/front 的定义、
以及三个平台是否经过统一的 platform-aware normalization。每条结论都给出代码证据或实测数据来源；
join 成功（token NPZ 与场景 JSONL 数值一致）只能证明两份数据同源，不能证明轴语义，因此不作为证据。

## 采用的约定（三平台一致）

统一场景文件 `3eed_scenes.jsonl` 中每个 scene 记录声明（`qa_pipeline/convert_3eed.py:232`）：

```
frame.convention = {"front": "+x", "left": "+y", "up": "+z"}
frame.origin     = "sensor"      # 原点在传感器/ego 位置
frame.units      = "m"
frame.right_axis = [0, -1, 0]    # 即 right = -y（convert_3eed.py:67-71）
frame.forward_axis = [1, 0, 0]
frame.up_axis    = [0, 0, 1]
```

- **front = +x**：右手系，+x 指向该帧的"前方"（见下方 ego/front 定义）。
- **left = +y**（因此 right = -y）。
- **up = +z**。
- 轴向命名一致性要求：判定关系时一律用 `dx = target.x - reference.x` 判 front/behind，
  `dy = target.y - reference.y` 判 left/right；绝不用"camera/image 坐标习惯"直接套 y 轴。

## 三个平台各自的 normalization（platform-aware）

`qa_pipeline/convert_3eed.py:210-241`：

| 平台 | frame 名 | 施加的变换 | 说明 |
|---|---|---|---|
| waymo | `waymo_rotated_sensor_view` | 绕 z 轴旋转 `view_angle_deg`（来自 `scan_id` 最后一个数字查 `WAYMO_VIEWS=[F,FL,FR,SL,SR]`） | 旋转后 +x 对准该帧所选 sensor view 的朝向 |
| drone | `drone_raw_sensor_frame` | z 坐标 +1.8 m（只动点云/box 高度，不动 x/y） | 原始水平坐标保持不变 |
| quad | `quad_raw_sensor_frame` | 无 | 原始水平坐标保持不变 |

点云需要同样变换才能和 box 同系，因此每个 scene 还写了 `frame.points_transform`
（waymo `{"rot_z_deg": angle}`，drone `{"translate_z_m": 1.8}`，quad `null`），下游加载原始点云时必须施加。

## ego / front 的定义

- 原点是传感器（`frame.origin = "sensor"`），不是车辆几何中心；对 drone/quad 就是机载传感器位置。
- **front（+x）= 该帧点云的"真实前方"**：
  - waymo：官方 loader 把点云和 GT box 一起旋转到所选 camera view 的朝向
    （`joint_det_dataset.py:708-714` 调 `transform_to_front_view(xyz, gt_bboxes[0], WAYMO_VIEWS[lidar_id])`），
    即 +x 是该 view 的视线/朝向方向，而不是车辆原始 heading。
  - drone/quad：保持 metadata 原始帧，+x 是传感器在记录时的朝向（机头方向）。
- 因此"front/left"是**帧内**（sensor-relative）概念，不是全局 (ego→world) 概念——这也是
  两平台都不应用 `pose` 的原因（见下）。

## 证据一：代码路径

1. `3EED/utils/eval_det.py:675`（IoU/取角点工具函数的 docstring）：
   `Coordinate system: X-forward/backward(l), Y-left/right(w), Z-up/down(h)`
   —— 评测代码本身把 +x 当作长度轴（前后），+y 当作宽度轴（左右），+z 向上。
   这与我们采用的约定一致，说明坐标系不是转换脚本发明的，而是沿用 3EED 自身定义。
2. waymo 旋转：`3EED/utils/transform_waymo.py` 的 `rotation_angles = {F:0, FL:-45, FR:45, SL:-90, SR:90}`，
   对点云做 `points @ R.T`、对 box 中心做同样旋转、并把 `rotated_boxes[:, 6] += angle_rad`（yaw 同步旋转）。
   loader 在 `joint_det_dataset.py:708-714` 用 `scan_id.split("_")[-1]` 选 view 后同时变换点云与 box；
   `convert_3eed.py` 的转换严格镜像这一路径（box 用 `rotate_xy` + yaw 同步，见其 `parse_box`/`entries` 逻辑）。
3. drone 高度：`joint_det_dataset.py:429` `xyz[:, 2] += 1.8  # NOTE drone dataset's z coordinate is lower`，
   box 侧在 `_get_3eed_target_boxes`（同文件 ~488 行）同样 `bbox[2] += 1.8`；转换器只镜像了 box 侧并写入
   `points_transform` 供点云侧使用。
4. quad 无变换：loader 中 quad 分支没有 pose 应用代码（对应行被注释掉，见 `joint_det_dataset.py:423-426`），
   转换器保持一致。

## 证据二：caption 关系词投票（实测，全量）

方法：把每条只含一个 left/right/front/behind 的 caption 当作一次投票，对四种轴约定在多种坐标系
解释下打分（`qa_pipeline/audit_3eed.py`，报告 `qa_pipeline/reports/audit_report.json`）。

| 平台 | 坐标系解释 | 最佳约定 | 得分 |
|---|---|---|---|
| waymo | 原始 metadata 帧 | x_fwd_y_left / x_back_y_left | 0.549（≈掷硬币） |
| waymo | **视角旋转后** | **x_fwd_y_left** | **0.9488** |
| waymo | 应用 pose | x_fwd_y_right | 0.5614 |
| drone | **原始 metadata 帧** | **x_fwd_y_left** | **0.9776** |
| drone | 应用 pose | x_fwd_y_left | 0.6276 |
| quad | **原始 metadata 帧** | **x_fwd_y_left** | **0.9906** |
| quad | 应用 pose | x_fwd_y_left | 0.6262 |

结论：y 轴方向（left=+y）由该投票决定；waymo 必须用**视角旋转后**的帧，drone/quad 用**原始帧**、
且都**不能应用 pose**（应用后得分掉到掷硬币水平附近）。
注意 voting 对 x 轴的前/后符号不敏感（x_fwd 与 x_back 同分，因为 captions 里 front/behind 样本极少），
x 轴方向由证据三决定。

## 证据三：相机投影 / 相关性（实测，与 extrinsic 假设无关）

`qa_pipeline/audit_projection.py`，报告 `qa_pipeline/reports/projection_report.json`：

- drone/quad：假设 extrinsic 把 box 变到相机系时，depth>0 比例 100%，box 中心落在标注 2D 框内
  80.5% / 83.3%（quad 用 xywh 口径 91.8%），光轴≈+x（drone 俯仰 15.3°，quad 精确 +x）
  → 前方是 +x；同时 image-right ≈ **-y**（1.0 一致率）→ right = -y、left = +y。
- waymo：extrinsic 两条方向假设都不成立（落框率 0%，深度中位数 2.9 m 不合理），原因未查明，
  因此 waymo 的结论不依赖该方法；另有不依赖 extrinsic 的 Pearson 相关法（2D 框中心 u/v 对 3D 坐标）
  给出 u vs y 旋转后 r=-0.9065，同样支持 left=+y。

## 未决 / 使用注意

- waymo 的 `image_extrinsic` 投影路径语义未查明（做可视化时需再查 `utils/box_util.py`）；不影响本约定。
- 由 voting 表可推：**waymo 的 "front" 是所选 sensor view 的朝向**，不同 view（F/FL/FR/SL/SR）的帧
  旋转角度不同；下游若混用 view，需以 scene 内的 `view_angle_deg` 为准。
- caption 中的空间词大量是"图像相对"表述（"left of center in the view"），与本文的 metric 帧方向
  是两回事；QA 标签只按本文约定生成，caption 词只用于泄露审计，不参与几何判定。

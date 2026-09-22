# 架构说明

## 1. 从单目标到双目标 Grounding

3EED 公开代码的主路径是一条 referring expression 监督一个目标。本项目把数据单位改为一对关联目标：

- `target 0`：当前 `ground_info` 的主目标。
- `target 1`：主目标 `others` 框中，能与同帧另一条 `ground_info` 按类别和 7D box 精确对齐的目标。

两个目标分别保留：

- 真实 caption 和 positive character span；
- GT 3D box 和 instance ID；
- 独立 Hungarian/contrastive 监督；
- 独立 decoder query 索引。

因此，这是同一帧中两个目标的联合监督，不是把单目标结果事后拼起来。当前公开 `others` 数据只能稳定构造两目标 pair；代码的张量路径支持可变 `target_count`，但仓库中的最终训练数据每条均为 2 个目标。

## 2. Grounding token

导出文件的主要形状是：

```text
object_token                 [batch, targets, 288]
contrastive_object_token     [batch, targets, 288]
query_index                  [batch, targets]
contrastive_query_index      [batch, targets]
pred_center                  [batch, targets, 3]
pred_size                    [batch, targets, 3]
gt_box                       [batch, targets, 7]   # 只用评测/诊断
```

`object_token` 是 soft alignment selector 选中的 decoder query feature；`contrastive_object_token` 是 contrastive selector 选中的同层 feature。v4 QA 使用 contrastive 版本，并保证 token、predicted center 和 predicted size 来自同一 query 索引。

token 已经经过点云编码、语言交互和 Transformer decoder，不是原始 PointNet++ visual token。

## 3. QA 后端

对象 A/B token 先加上可学习 role embedding，经过一层 8-head Transformer 交互，再使用共享 MLP 映射到 LLM hidden size：

```text
288D token
  + A/B/C role embedding
  → 1-layer, 8-head relation Transformer (288D)
  → Linear 288→1024 → activation → Linear 1024→3584
  → Qwen2.5-7B-Instruct embedding sequence
```

训练分两步：

1. 冻结 LLM，训练 relation module + projector。
2. 用最佳 projector checkpoint 初始化，训练 LoRA 和后端模块。

## 4. 显式几何的边界

GT box/center 用于：

- 根据几何自动生成 QA 正确答案；
- 计算 IoU 和 Grounding 成功率；
- 按 Grounding 正确/错误分组做私有诊断。

LLM 公开输入为：

```text
question text + implicit grounded object tokens
```

QA 输入不包含 GT/pred box、center、coordinate 或 distance 数值。

## 5. 三个及以上目标

导出器、mask 和 QA role 结构可扩展到三个及以上 token，但当前最终数据和实验只训练、验证了两目标。要宣称稳定输出三个以上目标 token，还需要有每个目标的文本和 GT box 的三目标标注，并重新训练与评测 joint accuracy。

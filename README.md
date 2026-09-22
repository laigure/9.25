# 3EED Multi-Object Grounding → Spatial QA

本仓库保存当前可复现的研究阶段成果：先在 3EED 单目标代码上完成双目标验证，再升级为每个场景 1–5 个目标的可变数量 Grounding；每个目标导出一个 288 维 decoder token，随后通过关系 Projector 和 Qwen2.5-7B LoRA 完成隐式 token 空间问答。

## 当前结论

- 多目标 Grounding 不是把两个 token 简单合并。每条数据有两个真实目标，每个目标有独立 caption、positive span、GT box、instance ID 和匹配 query。
- 最终 Grounding token 是 Transformer decoder/query feature，每个目标一个 288 维 token；两个目标使用不同 query 索引。
- QA 公开输入只含问题文本和隐式 token，不向 LLM 输入 box、center、coordinate 或 distance。GT 几何只用来生成标签和私有诊断。
- Grounding 正确条件下，v4 QA 准确率为 **80.00%**；打乱 token 降至 **36.91%**。
- 不使用 GT IoU 筛选的端到端 QA 准确率为 **57.08%**，多数类基线为 **35.29%**。

详细数字、失败实验和解释见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)。

## 场景级 v5（进行中）

- Grounding 数据已从固定双目标升级为每帧 **1–5 个**有真实 caption 的目标：train 2,701 帧/3,687 目标，val 2,708 帧/3,794 目标。
- 全部 7,481 条 caption/class/3D box 已逐项回查原始 `meta_info.json`；修正版 positive span 审计为 0 个复制错配、0 个明确环境物体词误选。这只能证明转换忠于发布标注，不能证明发布文字本身都正确；额外 RGB 抽查已发现至少一条疑似原始 caption 与目标框不一致。N=1…5 正例和负例见 [`qa_pipeline/artifacts/scene_multi_audit/EXAMPLES.md`](qa_pipeline/artifacts/scene_multi_audit/EXAMPLES.md)。
- RoBERTa、positive map 和 soft-token head 从 256 扩展到 512 token；真实 4 目标 GPU forward/loss/backward smoke 已通过。
- 新增 5,518 条高难一句话 QA，包含多干扰物关系、两步关系链、排序后推理和 1,470 条局部避障规划。
- 新严格评测要求关键词、全部三元组、标准完整句、A/B 交换、规划动作和 Grounding 同时正确，并按目标数 N=1…5 分别报告全部目标同时定位正确的准确率。
- 第一轮可变目标 Grounding 已在 4090 上完成 100 epoch。修正 v3 标注上的 contrastive Joint Acc@0.25 按 N=1/2/3/4/5 分别为 78.20%/59.51%/42.45%/0%/0%；它是含旧 span 训练噪声和严重目标数不平衡的 baseline，不能作为最终 v5 模型。上文 40.38%/80.00%/57.08% 仍属于旧双目标 v4。

## 目录

| 路径 | 内容 |
|---|---|
| `grounding/3EED/` | 3EED 源码快照及多目标修改 |
| `grounding/annotations/others_linked_data/` | 从 `ground_info[].others` 构建的 train/val 双目标标注 |
| `qa_pipeline/` | QA 数据生成、token 接入、Projector/LoRA 训练和评测代码 |
| `artifacts/local_qa/` | v2/v3 及早期 QA 数据、token 和诊断产物 |
| `artifacts/local_qa/scene_multi_data/` | 场景级 1–5 目标 Grounding 标注 |
| `artifacts/local_qa/scene_reasoning_qa_v1/` | 高难一句话 QA、局部规划和 oracle 评测 |
| `artifacts/remote/others_linked_tokens/` | 最终 Grounding train/val 全量 token 导出 |
| `artifacts/remote/qa_correct_others_v4_contrastive/` | 双目标均 IoU≥0.25 的 token 能力诊断 QA |
| `artifacts/remote/qa_all_others_v4_contrastive/` | 不做 IoU 筛选的端到端 QA |
| `artifacts/models/` | 关系 Projector 和 LoRA 最佳权重 |
| `artifacts/results/` | Grounding、打乱 token、A/B swap 和端到端评测 |

## 架构

```text
Point cloud + N referring expressions (N=1...5 in released data)
              │
              ▼
  multi-target 3EED decoder queries
              │
       token A ... token N (288D each)
              │
 role embedding + 1-layer/8-head relation Transformer
              │
        MLP 288 → 1024 → 3584
              │
      Qwen2.5-7B-Instruct + LoRA
              │
       natural-language spatial answer
```

完整张量、匹配和数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 快速检查

```bash
python tools/validate_release.py
```

该脚本检查关键文件、QA 行数、token 形状、模型权重和结果字段。

## 复现入口

1. 按 `grounding/3EED/README.md` 配置 3EED/Waymo 原始数据。原始点云和 RGB 未包含在本仓库。
2. 多目标数据审计与训练脚本在 `grounding/3EED/scripts_multi/`。
3. token 导出逻辑在 `grounding/3EED/utils/qa_token_export.py`。
4. QA v4 主入口是 `qa_pipeline/run_others_v4_relation.sh`，端到端分析是 `qa_pipeline/analyze_end_to_end.py`。

## 大 checkpoint

最终 Grounding checkpoint 大小 738,269,590 bytes，超过 GitHub 普通 Git 的 100 MB 单文件限制，因此未提交。其 SHA-256 为：

```text
84052fb17030fa0e8638e0fd9e91ec878afd9221c5293d1942c3a413711addac
```

仓库已包含该 checkpoint 导出的全部 train/val token、评测结果和重现代码。

## 数据来源

标注和 token 由 3EED/Waymo 数据衍生。使用者需同时遵守原始数据集和 3EED 代码的授权条款。本仓库不包含原始传感器数据。

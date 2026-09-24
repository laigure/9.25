# 3EED Multi-Object Grounding → Spatial QA

本仓库保存 3EED 多目标 Grounding、288 维 decoder token 导出、Pairwise Projector 和 Qwen2.5-7B LoRA 空间问答的可复现实验。当前正式方案只使用单条公开 caption 内明确提到的主目标和 `others`，不再拼接多个 caption。

## 当前结论

- 当前 Grounding 一条记录严格来自一个公开 `ground_info` caption，不拼接 caption，不使用 N=1；经文本—框歧义筛查后，正式数据为 train 644、val 714，目标数为 N=2 或 N=3。
- `others` 是上下文框集合，不能全部当作 caption 提到的目标。只有由同一句话明确引入、类别只有一个候选框且能得到不同非空间描述的目标才进入监督。
- 最终 Grounding token 仍是每个目标一个 288 维 Transformer decoder/query feature；同场景目标必须匹配不同 query。
- 新自然 QA 共 22,680 条，普通方位、距离、ego 运动和 object 运动各 5,670 条。问题不出现坐标、box、center、多选字母或 `Object A/B`，答案为一个完整标准句。
- GT 坐标只保存在 private GT，用于答案、严格评测和可选辅助关系损失；不拼接到 token，不进入问题、Qwen embedding 或推理。
- Grounding、Pairwise Projector 和 Qwen LoRA 的 GPU 阶段尚未启动：当前远程实例没有 `/dev/nvidia0`。可恢复脚本会在 GPU 恢复后从 Grounding 训练继续。

详细数字、失败实验和解释见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)。

完整新数据见 [`qa_pipeline/artifacts/native_single_caption_v1/`](qa_pipeline/artifacts/native_single_caption_v1/)，逐项记录见 [`qa_pipeline/WORK_LOG.md`](qa_pipeline/WORK_LOG.md)。

## 历史场景级拼接实验（保留追溯）

- Grounding 数据已从固定双目标升级为每帧 **1–5 个**有真实 caption 的目标：train 2,701 帧/3,687 目标，val 2,708 帧/3,794 目标。
- 全部 7,481 条 caption/class/3D box 已逐项回查原始 `meta_info.json`；修正版 positive span 审计为 0 个复制错配、0 个明确环境物体词误选。这只能证明转换忠于发布标注，不能证明发布文字本身都正确；额外 RGB 抽查已发现至少一条疑似原始 caption 与目标框不一致。N=1…5 正例和负例见 [`qa_pipeline/artifacts/scene_multi_audit/EXAMPLES.md`](qa_pipeline/artifacts/scene_multi_audit/EXAMPLES.md)。
- RoBERTa、positive map 和 soft-token head 从 256 扩展到 512 token；真实 4 目标 GPU forward/loss/backward smoke 已通过。
- 已生成 5,518 条无直接文本泄漏的一句话 QA，包含多干扰物关系、两步关系链、排序后推理和 1,470 条局部避障规划；公开问题不再出现原始 caption、坐标或 left/right/front/behind 答案提示词。
- 新严格评测要求关键词、全部三元组、标准完整句、A/B 交换、规划动作和 Grounding 同时正确，并按目标数 N=1…5 分别报告全部目标同时定位正确的准确率。
- 第一轮可变目标 Grounding 已在 4090 上完成 100 epoch。修正 v3 标注上的 contrastive Joint Acc@0.25 按 N=1/2/3/4/5 分别为 78.20%/59.51%/42.45%/0%/0%；它是含旧 span 训练噪声和严重目标数不平衡的 baseline，不能作为最终 v5 模型。上文 40.38%/80.00%/57.08% 仍属于旧双目标 v4。
- 无泄漏 QA v2.1 的 Projector→LoRA 训练已完成。完整 val 2,862 条上，LoRA 的关系关键词/严格三元组/标准句/规划动作/swap/严格语言合取/端到端分别为 **45.91%/34.21%/32.98%/44.01%/20.77%/24.67%/13.98%**；同类别打乱 token 后严格语言合取降至 **10.52%**、端到端降至 **5.66%**。模型使用了正确 token，但多目标关系和交换一致性仍是主要瓶颈。

## 目录

| 路径 | 内容 |
|---|---|
| `grounding/3EED/` | 3EED 源码快照及多目标修改 |
| `grounding/annotations/others_linked_data/` | 从 `ground_info[].others` 构建的 train/val 双目标标注 |
| `qa_pipeline/` | QA 数据生成、token 接入、Projector/LoRA 训练和评测代码 |
| `qa_pipeline/artifacts/native_single_caption_v1/` | 当前原生单-caption Grounding PKL、22,680 条 QA、private GT、审计、样例和 SHA-256 |
| `artifacts/local_qa/` | v2/v3 及早期 QA 数据、token 和诊断产物 |
| `artifacts/local_qa/scene_multi_data/` | 场景级 1–5 目标 Grounding 标注 |
| `artifacts/local_qa/scene_reasoning_qa_v1/` | 已废弃的直接文本泄漏版本，仅供追溯 |
| `artifacts/local_qa/scene_reasoning_qa_v2_noleak/` | 仅角色文字 + 隐式 token 的场景 QA、私有 GT、审计和样例 |
| `artifacts/remote/others_linked_tokens/` | 最终 Grounding train/val 全量 token 导出 |
| `artifacts/remote/qa_correct_others_v4_contrastive/` | 双目标均 IoU≥0.25 的 token 能力诊断 QA |
| `artifacts/remote/qa_all_others_v4_contrastive/` | 不做 IoU 筛选的端到端 QA |
| `artifacts/models/` | 关系 Projector 和 LoRA 最佳权重 |
| `artifacts/results/` | Grounding、打乱 token、A/B swap 和端到端评测 |
| `artifacts/results/scene_qa_v2_noleak/` | 无泄漏场景 QA 的 Projector、LoRA、打乱 token 严格结果 |

## 架构

```text
Point cloud + one released caption with N grounded noun spans (current N=2...3)
              │
              ▼
  multi-target 3EED decoder queries
              │
       token A ... token N (288D each)
              │
 all directed token pairs + 2-layer/8-head relation Transformer
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
2. 当前完整入口是 `qa_pipeline/run_native_multi_natural_qa.sh`；它会构建原生单-caption 数据、训练 Grounding、导出 token，并运行有/无辅助关系损失的两组 Projector + LoRA。
3. token 导出逻辑在 `grounding/3EED/utils/qa_token_export.py`。
4. 历史 QA v4 入口为 `qa_pipeline/run_others_v4_relation.sh`，仅用于追溯旧结果。

## 历史大 checkpoint

最终 Grounding checkpoint 大小 738,269,590 bytes，超过 GitHub 普通 Git 的 100 MB 单文件限制，因此未提交。其 SHA-256 为：

```text
84052fb17030fa0e8638e0fd9e91ec878afd9221c5293d1942c3a413711addac
```

仓库已包含该历史 checkpoint 导出的 train/val token、评测结果和重现代码；当前原生单-caption checkpoint 需等待 GPU 训练完成。

## 数据来源

标注和 token 由 3EED/Waymo 数据衍生。使用者需同时遵守原始数据集和 3EED 代码的授权条款。本仓库不包含原始传感器数据。

# 3EED 多目标 Grounding 下一阶段任务清单

## 总任务

复现论文正式 multi-object grounding，确认一句话中的多个目标如何与多个 decoder query/token 对应，并为后续空间 QA 提供可靠的隐式目标表示。

完成条件：

1. 在正式或等价的多目标标注上完成训练和评测。
2. 指标计算方式与论文 Table 3 一致。
3. 能为一条多目标表达导出多个目标 token、预测框、置信度和匹配关系。
4. 在不向 LLM 输入坐标的条件下，用正确预测的目标 token 跑通一批 QA 样例。

---

## 阶段 A：确认正式多目标数据

- [x] A1. 搜索官方仓库、数据下载包和服务器，寻找 `waymo_multi_train_info.pkl`、`waymo_multi_val_info.pkl`。（已找到）
- [x] A2. 检查论文附录、补充材料和项目主页中的 multi-object 数据生成说明。（见 `OFFICIAL_MULTI_ANNOTATION_SEARCH.md`）
- [x] A3. 如果官方文件未公开，整理缺失文件、预期字段和代码入口，准备向作者询问。（官方文件已找到，无需询问）
- [x] A4. 读取官方 multi loader，列出每个样本必须包含的字段。
- [x] A5. 编写数据检查脚本，统计句子数、每句目标数、类别、框数量和异常样本。

产出：`MULTI_DATA_AUDIT.md`、数据统计 JSON、可用的 multi train/val 标注。

通过标准：每条多目标表达都能稳定读出两个或更多 GT 框，表达中的目标片段与框能对应。

---

## 阶段 B：严格复现论文多目标训练

- [x] B1. 固定官方代码版本、配置、checkpoint、随机种子和数据划分。
- [x] B2. 先运行 20 至 50 条样本的过拟合测试，确认多目标 loss 能下降。
- [x] B3. 检查 Hungarian matching 或目标分配是否把多个 GT 分给不同 query。
- [x] B4. 启动完整 multi-object 训练并保存每轮 checkpoint 与日志。（200 epochs 完成）
- [x] B5. 使用官方评测脚本计算 Acc@0.25、Acc@0.5、mIoU。（JointAll-bbf@0.25 40.62%、@0.5 18.46%、mIoU 34.81%；另有新口径逐目标 acc@0.25 59.5%）
- [ ] B6. 与论文 Table 3 对照，分析类别和目标数量上的误差。

产出：训练配置、日志、checkpoint、正式评测结果。

通过标准：评测代码与论文口径一致；结果明显超过当前自制数据的 5.79%，并尽量接近论文报告值。

---

## 阶段 C：确认多目标 token

- [x] C1. 定位 decoder 最后一层 query feature 的代码位置和张量形状。（`last_query_features`，288 维）
- [x] C2. 确认每个目标使用哪个 query，以及选择过程是否依赖 GT 文本片段或 GT 框。（soft-token 文字 span 分数 + 唯一分配，不依赖 GT 框）
- [x] C3. 为每条表达导出所有预测 query 的 token、预测框、分数和目标匹配结果。
- [x] C4. 验证不同目标是否对应不同 token，是否出现重复 query 指向同一目标。（`verify_multi_export.py` 全量验证）
- [x] C5. 分别测试三种 query 选择方法：GT 匹配、预测分数选择、语言目标片段选择。（soft-token vs contrastive 已比较；GT 匹配只作离线诊断）
- [x] C6. 做 token 探针：只用 token 预测目标类别、相对方向和距离区间，判断 token 含有什么信息。（线性探针：val 选对目标子集 R² x=0.900/y=0.792，|Δy|≥2m 左右排序 95.7%，见 `OTHERS_MULTI_QA_REPORT.md` 第 5 节）

产出：`multi_token_export/*.pt`、token 元数据 JSONL、token 分析报告。

通过标准：推理阶段不使用 GT 框，也能为多个目标选择对应 token；明确 token 维度、层号、数量及其包含的空间信息。

---

## 阶段 D：接入空间 QA

- [x] D1. 先筛选 Grounding 正确的多目标样本，作为受控 QA 集。（train 2447 / val 620 题）
- [x] D2. 用 GT 几何标注生成问题和答案，禁止使用预测框生成答案标签。
- [x] D3. LLM 输入只包含多个 grounding token、问题文本和必要的角色标记，不输入坐标、中心和尺寸。
- [x] D4. 训练 projector，将 288 维 grounding token 映射到 LLM hidden dimension。
- [x] D5. 先训练简单关系：left/right、front/behind、near/far。（val：总 62.26%，closer 100%，方向题仍弱——见 `OTHERS_MULTI_QA_REPORT.md`）
- [ ] D6. 再训练多目标关系、危险目标识别和简单路线决策问答。（多目标关系已训；危险目标/路线决策待做）
- [ ] D7. 分别评测 GT 匹配 token、正确预测 token、全部预测 token，分离各模块误差。（正确预测 token 已评；全部预测含失败样本的端到端评测未做）

产出：QA 数据 JSONL、projector checkpoint、LoRA checkpoint、最小演示程序和分阶段评测表。

通过标准：正确 token 条件下，模型能稳定生成完整空间关系回答；完整预测条件下能量化 Grounding 错误造成的下降。

---

## 阶段 E：扩展到路线规划

- [ ] E1. 定义 ego vehicle、可行驶区域、障碍物和目标位置的场景表示。
- [ ] E2. 从连续帧或轨迹标注生成危险目标与未来路径问题。
- [ ] E3. 先输出离散决策：直行、减速、停止、左绕、右绕。
- [ ] E4. 有可靠轨迹 GT 后，再输出路径点或轨迹 token。
- [ ] E5. 增加碰撞率、到达率、路径长度和决策正确率评测。

产出：规划 QA 子集、规划评测脚本和演示。

通过标准：模型的规划答案有几何 GT 或轨迹 GT 支撑，并通过独立安全规则检查。

---

## 当前状态（2026-09-22 更新）

阶段 A/B/C 与 D1–D5 已完成，D6/D7 部分完成，阶段 E 未开始。完整数据、口径与诊断见 `OTHERS_MULTI_QA_REPORT.md`。

紧接着要做的两件事：

1. 判定方向题弱项是欠训练还是结构问题（10 轮对照训练 `runs/others_lora_e10` 在跑；若 10 轮仍不涨，用几何预训练 projector 或几何辅助损失）。
2. B6 与论文 Table 3 的正式对齐（本机只有单目标 80.67% 与旧联合口径，Table 3 的 others-multi 数值需核对论文）。

## 当前禁止混用的结果

- 论文 Table 3 正式 multi-object 指标。
- 公开默认单目标 Grounding 指标。
- 本项目自制 908 条拼接数据的 5.79% 原型结果。

三者必须分别记录数据、checkpoint、评测脚本和实验名称。

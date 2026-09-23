# 隐式 Grounding QA 数据集与 LoRA 训练操作记录

更新时间：2026-09-20。每完成一个可复查动作，立即在这里追加：做了什么、结果、产物、下一步。不要记录远程密码。

## 目标和边界

- 训练输入：点云经 Grounding 得到的隐式目标/场景特征 + 自然语言问题；不向 LLM 输入 GT/pred box、中心、坐标、距离或预计算关系。
- GT 几何只用于离线生成 QA 答案、筛选边界样本和评测。
- 当前 3EED 可验证的空间 QA 先做成数据集并训练 LoRA；需要未来轨迹/道路信息的驾驶规划题单列为待补标注，不用示意文本伪造训练答案。
- 训练/验证/测试须按完整驾驶序列隔离。此前按帧切分的数据和指标仅作探索对照。

## 操作记录

### 2026-09-20 / 01 — 建立持续记录

- 已创建本文件，固定目标、输入限制和记录格式。
- 接下来检查本地代码、已有场景/QA/特征产物和远程主机状态，确定可直接复用与必须重做的部分。

### 2026-09-20 / 02 — 盘点现有代码与远程环境

- 本地已有 3EED 场景转换器、旧版 Waymo QA、token bank 和 `train_qa.py`。旧版 QA 只用 3EED val，按帧随机分组，完整 caption 的空间词可能进入 token；不能直接当作本次最终训练集。
- 远程 `/root/3eedqa/3EED` 与数据盘可用，RTX 4090 当前空闲。旧 LoRA 与 projector 产物保留在 `/root/autodl-tmp/3eed_data/qa_v1/runs/`。
- `train_qa.py --geom none` 能只用 288 维隐式 token；其 `--geom center/box/relative` 路线不用于本次目标。
- 下一步：抽样审计无空间词的指代短语，设计重新导出 token 的策略；用原始 Waymo train/val 序列分别构造严格隔离的数据。

### 2026-09-20 / 03 — 确定可用标注规模与指代清理方向

- 远程统一场景文件包含 Waymo train 2,701 帧/94 个序列、val 2,708 帧/93 个序列；分别有 843/910 帧包含至少两个带 caption 的对象。drone/quad 没有同帧双 caption，不直接纳入双目标题。
- 在旧 Waymo val token bank 的 3,782 条描述上，现有 `short_expression()` 初步提取后，保守空间词规则保留约 1,009/1,300 个无序双目标 pair（其中 295 个跨类别）；这只是文本层预审，还需重新跑 Grounding 和 IoU 检查。该规则会把“front grille”“right hand”等外观短语也标出，后续需区分外观词与关系线索。
- 决定制作新版独立 token bank：用去空间关系的短指代重新运行 Grounding；保留旧 bank 和旧训练结果作对照，不覆盖。
- 下一步：实现中性指代提取与新版 token 导出，先做小规模 pilot 测真实定位质量，再全量导出。

### 2026-09-20 / 04 — 实现中性指代与重新导出脚本

- 新增 `neutral_query.py`：从原 caption 抽取简短目标描述，保守过滤方向/位置/视图词；以描述内出现的第一个类别名定位文本 span，选择器不依赖 GT box 或 GT category。
- 新增 `export_neutral_tokens.py`：重新运行公开 3EED checkpoint，导出 288 维最后一层 query token、预测框、GT 对照与 IoU；`--limit-per-split` 可先抽样。GT 类别仅用于离线核验描述所指名词和标签，不进入 query 排序或 LLM 输入。
- 两个脚本通过 Python 语法检查；在旧 object bank 的样本上做了中性短语与类别名解析烟测。
- 下一步：同步远程并对 train/val 各做少量抽样推理，确认脚本输出、过滤率和中性描述定位质量。

### 2026-09-20 / 05 — 远程同步与首轮 pilot 修复

- 已把 `neutral_query.py`、`export_neutral_tokens.py` 复制到 `/root/3eedqa/qa_pipeline/`，远程语法检查通过。
- 首次执行 32+32 样本 pilot 时在导入阶段停止：远程 `qa_pipeline/` 缺少本地已有的 `build_qa.py`，`neutral_query.py` 的短描述函数无法导入。没有生成任何 token，也没有启动训练。
- 下一步：把依赖文件同步到远程，重跑同一 pilot；失败原因和修复动作保留在记录中。

### 2026-09-20 / 06 — 中性指代 pilot 成功与质量判断

- 已同步 `build_qa.py`，重跑 train/val 各随机 32 条 pilot，成功生成 `/root/autodl-tmp/3eed_data/qa_v2_pilot/` 的 64×288 token、bank 和统计。
- 原 loader 可见的 Waymo 标注数：train 3,682、val 3,782；保守中性化后可用：train 3,048、val 3,082（全量推理前的候选数）。
- pilot 定位 IoU≥0.25：train 0.5625、val 0.5312；IoU≥0.5：train 0.4062、val 0.3125。明显低于使用原完整 caption 的旧基线，说明移除位置提示后 Grounding 更困难；新版结果必须如实报告，不能把旧 baseline 指标当成本次性能。
- 下一步：全量导出约 6,130 个中性 token，并按序列、目标唯一性和 QA 场景构造新数据；LoRA 不使用任何显式几何输入。

### 2026-09-20 / 07 — 中性 token 全量导出完成

- 远程 `/root/autodl-tmp/3eed_data/qa_v2/` 已保存 6,130×288 token 和 `neutral_bank.jsonl`，全量进程正常结束。
- train：3,048 条，IoU≥0.25 为 0.5728，IoU≥0.5 为 0.3842；val：3,082 条，分别为 0.4750、0.3021。这是去除空间提示后的真实预测质量，后续 QA 测试不能按预测 IoU 筛选样本。
- train 2,337 帧中 609 帧有至少两个中性对象、96 帧至少三个；val 分别为 2,344、619、90。可制作单目标、双目标与少量三目标问题。
- 下一步：实现按序列隔离的 QA builder，输出公开输入/答案和单独的 GT 私有标签，完成泄漏与分组审计。

### 2026-09-20 / 08 — 分场景 QA 数据集生成

- 新增 `build_scenario_qa.py`。首次远程运行因 object_id 含 `#` 而触发过严的格式断言，修正后重新运行成功；这次失败未写出有效 QA。该修复已同步到本地脚本。
- 远程数据输出：`/root/autodl-tmp/3eed_data/qa_v2/scenario_qa/{qa.jsonl,private_gt.jsonl,summary.json}`。共 6,984 条：train 3,364（单目标 2,629/双目标 655/三目标 80），val 1,980（1,463/462/55），test 1,640（1,307/304/29）。
- 切分按 94/46/47 个驾驶序列完成，三组序列零交集。问答只引用中性描述和隐式 token；GT box 与答案规则只在 `private_gt.jsonl`。没有按预测 IoU 过滤测试样本。
- 三目标暂只做“哪个目标离参照物更近”。3EED 无未来轨迹/可行驶区域真值，因此路线规划题不生成训练标签。
- 下一步：为可变数量目标 token 接入现有 Qwen projector+LoRA，验证模型实际 prompt 不携带 GT 关系与坐标，然后训练/评测。

### 2026-09-20 / 09 — LoRA 接入、烟测和正式训练启动

- 新增 `train_scenario_qa.py`，沿用 Qwen2.5-7B-Instruct、288→1024→3584 projector、r=16 LoRA；覆盖输入拼接函数，只把问题文字和 1–3 个 288 维隐式目标 token 送入模型。验证 token/object_id 一一对应。模型输入函数不读取 `private_gt.jsonl`、box、center、距离或答案规则。
- 远程已同步训练脚本及其依赖，Python 语法检查通过。64 条训练/24 条验证、1 epoch 烟测成功保存 checkpoint 并完成生成；小样本准确率 0.0833，仅说明链路能运行，不代表方法性能。
- 审核问答例子时发现问题文字中的 `Object A is The pedestrian` 不自然，以及三目标答案偏向 B（72/80）。已改成 `Object A: ...`，并用稳定哈希对半交换三目标候选次序；重新生成 6,984 条数据。当前三目标 train A/B=33/47、val=29/26、test=17/12。
- 已启动正式训练进程 PID 26128，日志 `/root/autodl-tmp/3eed_data/qa_v2/runs/full_lora/train.log`；设置 3 epochs、batch 2、累积 4、学习率 1e-4。将输出 best/last checkpoint 和完整 val/test 生成结果。训练期间继续审计数据及等待结果。

### 2026-09-20 / 10 — 数据泄漏与多数类基线审计

- 远程重新读取 6,984 条公开 QA/私有 GT：qa_id 全部唯一；所有 object description 均无 left/right/front/behind/near/far/center/distance 等显式空间词；公开 QA 不含 gt_box/gt_center/pred_center/pred_size/distance/answer_key 字段。
- 按训练集每种题型最常见答案预测，验证总体准确率 0.3530，测试 0.4195；题型验证基线依次为 ego 0.3828、relative 0.2446、closer 0.4727。正式模型应与此基线比较，不能只报告总准确率。
- 下一步：等待正式训练、检查 val/test 按题型准确率及模型输出，再判断是否需要修正数据或训练方案。

### 2026-09-20 / 11 — 首轮训练与 token 因果对照准备

- 正式 LoRA 第 1 epoch 完成，用时 257.0 秒，验证集前 400 条的 loss=0.0921；训练继续进行。loss 偏低可能反映答案模板短且类别不均衡，须等待生成准确率与对照实验，不据此声称有空间推理能力。
- 新增 `make_shuffle_control.py`。远程已生成同类别、同官方 train/val 划分内打乱的 token bank，6,130 条中 6,124 条特征发生改变，同时保持 object_id 索引不变。训练结束后用同一个 checkpoint 评估该对照，检验模型是否依赖目标 token。
- 新增 `SCENARIO_QA_DATASET.md` 作为数据卡，记录数据流、题型、数量、命令和局限，并核对 Waymo +x 前/+y 左的官方坐标约定。

### 2026-09-20 / 12 — 数据产物同步到本地工作区

- 通过已配置的 SSH 密钥从远程复制 `scenario_qa/qa.jsonl`、`private_gt.jsonl`、summary、`neutral_bank.jsonl`、导出统计和 288 维 token NPZ 到本地 `qa_pipeline/artifacts/qa_v2/`。没有复制远程密码或凭据。
- 本地重新加载检查：公开 QA 6,984 条，token 数组形状 `(6130, 288)`，首条题目可正常解码。用户现在可直接在本地打开数据样本；远程正式训练继续运行。

### 2026-09-20 / 13 — 正式训练第 2 轮完成

- 第 2 epoch 用时 254.5 秒；验证集前 400 条 loss 从 0.0921 降到 0.0905。训练已进入第 3 epoch，仍使用全部 3,364 条训练 QA。
- 目前没有生成准确率，尚不能判定模型优于多数类或确实使用 Grounding token；后续完整 val/test 与打乱 token 对照是必要检查。

### 2026-09-20 / 14 — 三轮 LoRA 完成，进入生成评测

- 3 个 epoch 均完成：验证 loss 依次为 0.0921、0.0905、0.0915；最佳 checkpoint 来自第 2 轮，已自动重新加载。`best.pt` 和 `last.pt` 均保存于远程 `qa_v2/runs/full_lora/`。
- 训练进程尚未结束，现正对完整 1,980 条 val 生成回答，之后自动生成 1,640 条 test。不能把第 3 轮 loss 上升隐去；评测以最佳第 2 轮模型为准。
- 已将最佳 `best.pt`（约 35 MB，仅 projector 与 LoRA 权重，不含 Qwen base）和 `history.json` 复制到本地 `qa_pipeline/artifacts/qa_v2/runs/full_lora/`，避免远程实例中断导致本次训练丢失。

### 2026-09-20 / 15 — 完整验证结果与失败原因初探

- 最佳第 2 轮模型已生成完整 1,980 条 val：总准确率 0.3601，完整句精确匹配 0.3601，格式可解析率 1.0。按题型：ego 0.3978、relative 0.2316、closer 0.4364；训练集多数类基线分别为 0.3828、0.2446、0.4727。当前结果仅略高于总体多数类，双/三目标更差，不能宣称实现可靠空间推理。
- 输出分布塌缩：ego 1,463 条中 1,385 条答 `right_front`；relative 462 条中 408 条答 `right_behind`；closer 55 条中 49 条答 B。低 loss 主要反映固定答案模板容易学习，不能代表几何判断正确。
- 同一 288 维 token 的离线线性探针（仅 train 拟合）显示：ego train/val/test=0.7398/0.5284/0.5585，relative=0.9969/0.3810/0.3717，closer=1.0/0.3818/0.4138。ego token 至少含可解码位置线索，但当前 LoRA+projector 没有充分利用；双/三目标数据更少且泛化较弱。线性探针是诊断，不是最终 QA 模型。
- 主进程已转入 1,640 条 test 生成。下一步比较测试结果与打乱 token 对照，决定下一版训练方式。

### 2026-09-20 / 16 — 完整测试完成并备份结果

- 正式训练与 val/test 生成进程正常结束。test 1,640 条：总准确率 0.4201、完整句精确匹配 0.4201、格式可解析率 1.0；ego 0.4422、relative 0.3322、closer 0.3448。训练集多数类测试总基线 0.4195，当前模型几乎没有超越它。
- 已将 `eval_val.json`、`eval_test.json`、`train.log` 复制到本地 `qa_pipeline/artifacts/qa_v2/runs/full_lora/` 并重读验证数量和统计。所有结果都保留，不能只展示单目标或训练 loss。
- 已构造固定种子的 355 条 val 探针子集（ego 150、relative 150、closer 55），该子集正常 token 的现有 checkpoint 准确率为 0.3408。正在用同一个 checkpoint + 同类别打乱 token 生成对照，进程 PID 27282，日志 `qa_v2/runs/shuffle_probe/eval.log`。

### 2026-09-20 / 17 — token 打乱对照完成与本轮结论

- 同一最佳 checkpoint 在 355 条固定 val 探针上：正常 token 准确率 0.3408（121/355），同类别同官方划分打乱 token 后 0.3324（118/355）。按题型正常→打乱：ego 63/150→59/150、relative 34/150→33/150、closer 24/55→26/55。仅 19/355 条预测类别改变。
- 因而本轮模型主要依赖答案先验，几乎不使用 Grounding token；不能把 6,984 条数据、训练收敛或可解析的答案格式视作空间推理成功。该负结果保留，不删除 checkpoint。
- 已把探针 QA/GT 和打乱结果复制到本地 `qa_pipeline/artifacts/qa_v2/{probe_val,runs/shuffle_probe}/`。本轮请求的建集、LoRA 训练、完整 val/test 与 token 依赖性评测均已完成。
- 下一轮应在**仍只输入隐式 token**的前提下，先对训练集做题型/答案平衡，再用有监督的空间关系目标训练 token projector 或空间编码器；同时比较单目标 token 的线性探针与 LLM，避免仅凭固定句式 loss 选择模型。三目标需更多带不同类别与外观指代的同帧标注；规划需另外引入时序自车轨迹、地图和动态目标监督。

### 2026-09-20 / 18 — 用户要求扩充数据，先盘点真实标注容量

- 新请求：增加数据量，并保持上一轮“仅隐式 Grounding token 输入”的约束。先盘点现有中性 token bank：6,130 个独立目标标注、4,681 个帧；1,228 帧有至少两个中性目标、186 帧至少三个；同帧对象两两组合 1,708，三元组合 300。
- 训练侧 3,048 个目标、约 810 个可区分描述的对象 pair、96 个三目标帧；验证/测试来源侧 3,082 个目标、约 883 个 pair、90 个三目标帧。即使生成更多问题，独立帧/目标数不增加，必须在数据卡中同时列出。
- 离线可制标的新语义任务：单目标左右方位、近/远距离等级；两目标反向关系、左右先后、谁离自车更近；三目标更换参照目标的距离比较。全部答案仍由 GT 几何离线产生，问答输入只引用原中性 token，不加入坐标。先统计清晰样本数量并设边界间隔，再生成 v3，不覆盖 v2。

### 2026-09-20 / 19 — v3 扩充为 18,274 条并完成本地审计

- 新增 `build_scenario_qa_v3.py`，在 v2 三种题型上增加单目标左右/近远、双目标反向相对位置/谁更靠左/谁离自车更近、三目标轮换参照物/三个中谁离自车最近。答案仍由 GT box 离线计算；只沿用旧的 288 维中性 token。所有题目设置明确距离/方位间隔，回避边界含糊样本。
- 本地输出 `qa_pipeline/artifacts/qa_v3/scenario_qa/`：总 18,274，train 8,902、val 5,153、test 4,219；八个场景题型。与 v2 使用相同的 94/46/47 序列划分，不覆盖旧数据和模型。
- 真实独立信息量仍为 6,130 个带中性描述的目标、4,681 帧、1,228 个多目标帧、186 个三目标帧。扩充的是问答监督视角，不是新增采集场景；这一点在 summary 中单列。
- 审计通过：18,274 个 qa_id 唯一，train/val/test 序列零交集，公开 QA 无显式 box/center/距离/答案类别字段，中性目标描述无方向/距离词。训练集多数类基线 val 0.4153、test 0.4646。
- 已更新 `train_scenario_qa.py` 的答案解析器以支持新增题型。下一步：远程同步 v3 数据与脚本，先做小规模训练烟测，再进行 LoRA 训练并检查是否比 v2 更依赖 token。

### 2026-09-20 / 20 — v3 同步、烟测与正式 LoRA 启动

- 已将 v3 生成脚本、更新后的训练脚本、18,274 条 QA 与私有 GT 传至远程 `/root/autodl-tmp/3eed_data/qa_v3/scenario_qa/`；继续引用 v2 中同一份中性 token NPZ，不重新计算或引入显式几何输入。
- 远程 Python 语法检查通过。64 条 train、每题型最多 8 条 val 的 1 epoch 烟测完成 checkpoint 保存与答案生成，说明新增题型训练/解析链路可运行；小样本分数不作为方法性能。
- 启动正式 v3 LoRA 训练 PID 27868：全部 8,902 条训练问答，2 epochs，batch 2、累积 4；每题型固定抽取最多 50 条 val 作为快速 checkpoint 选择和训练后小规模生成。日志 `/root/autodl-tmp/3eed_data/qa_v3/runs/full_lora/train.log`。训练完成后仍需完整 val/test 与打乱 token 对照，不能仅凭小规模验证宣布改进。

### 2026-09-20 / 21 — 区分来源容量与实际入题容量

- 复核 v3 统计时发现初版 summary 的 `independent_scenes=4681` 指的是**中性 token 来源帧**，不能直接当成“实际进入 QA 的帧”。已修改构建脚本与 summary，分别报告来源 4,681 帧/6,130 目标和实际入题 4,519 帧/5,961 目标；训练集实际入题 2,250 帧/2,957 目标。
- 重新生成的 18,274 条 QA 数量与题型分布不变，修正 summary 已同步远程；数据卡对应更正。本次只修复统计定义，不影响正在运行的 LoRA 训练输入。
- 新增 `analyze_scenario_eval.py`，在后续结果中同时给出按 QA 行、按场景平均和训练集多数类基线，避免多题同帧造成的总分误读；先用 v2 测试结果验证脚本可用。
- 另外导出本地 `qa_pipeline/artifacts/qa_v3/EXAMPLES.txt`，每种题型各两条真实训练样例，便于人工核对问答形式；抽样检查了不同题型的问题和完整句答案。

### 2026-09-20 / 22 — 设置断线后继续评测

- 新增并同步 `run_v3_posttrain.sh`，远程语法检查通过，后台监视进程 PID 28145，日志 `/root/autodl-tmp/3eed_data/qa_v3/runs/posttrain.log`。
- 它会等正式训练 PID 27868 结束且 `best.pt` 存在后，自动使用最佳 LoRA 对完整 val/test 生成答案，然后在每题型最多 50 条 val 上做同类别打乱 token 对照。即使当前会话/额度中断，训练后的评测仍会继续并写盘；后续仍需人工读取结果、做场景级分析和同步本地。

### 2026-09-20 / 23 — 核对扩大独立场景的可行来源

- 现有统一场景文件还包括 drone 7,098 帧、quad 7,860 帧，Waymo 5,409 帧。它们有大量 GT 对象框，但 drone/quad 的同帧多目标**中性指代与相应 token**目前不足，且自车左右/前后语义与道路车坐标体系不相同，不能直接混进当前 Waymo 驾驶 QA 并声称是独立同分布样本。
- v3 此轮只从 Waymo 已导出的、可核验的 Grounding token 扩展语义任务；若下一步要真正增加独立场景，需要补充新的 Waymo 时序/多目标标注并重新导出 token，不能靠重复措辞实现。

### 2026-09-20 / 24 — 新题型 token 可解码性诊断

- 新增 `probe_v3_tokens.py`，仅在 train 序列拟合线性分类器，输入只用同一份隐式 288 维 token，不给坐标；完整数值保存本地 `qa_pipeline/artifacts/qa_v3/probe_tokens.json`。
- 单目标 token 的 val/test 线性探针：`ego_near_far` 0.8937/0.8748，`ego_side` 0.6942/0.7053，`ego_location` 0.5284/0.5585。说明新增单目标近远/左右任务至少有可从 token 解码的信号，适合检查 LoRA 是否真的学会使用 token。
- 双目标 `relative_location` val/test 仅 0.3009/0.2204，三目标最近物 0.2174/0.3704；多目标空间关系仍是明显难点。探针小样本训练准确率很高而跨序列泛化弱，不应视为最终模型成绩。

### 2026-09-20 / 25 — v3 LoRA 第一轮完成并备份

- 8,902 条训练 QA 的第 1 epoch 完成，用时 654.6 秒；每题型最多 50 条的 373 条验证子集 loss=0.0776。第 2 epoch 继续运行，尚无正式生成准确率。
- 已把第 1 轮 `best.pt`（projector+LoRA，不含 Qwen base）额外复制到本地 `qa_pipeline/artifacts/qa_v3/runs/full_lora/best_epoch1.pt`。第二轮若改进，最终最佳 checkpoint 将再同步；此备份避免实例意外中断时两轮训练全部丢失。

### 2026-09-20 / 26 — 回答“token 是否靠谱”：定位质量与选择器复核

- 核对 3EED `bdetr.py` 与导出脚本：`last_query_features` 确实是在最后一层 Transformer decoder 输出后、预测头之前的 288 维语言交互 query；根据 64 维 contrastive projection 对文字 span 打分选 query，与官方 evaluator 的 softmax/温度 0.07/文本 span 求和规则一致。不是误用了原始 PointNet token 或 64 维投影。
- 当前 v2/v3 token 的具体**选择 span 只覆盖第一个类别名词**，虽然模型读取整句中性描述，排序时颜色/服饰等区分同类目标的词没有直接参加 span 求和。这是潜在误选来源，需和“完整中性短语 span”做同样本对比，不能仅凭推测宣称修复。
- 根据旧 token bank 的 GT 框离线检查，v3 val 中所有输入目标均 IoU≥0.25 的比例：单目标 1,505/3,282=45.9%，双目标 155/1,638=9.5%，三目标 0/233。这解释了多目标 QA 为何可能受到严重 Grounding 噪声影响；GT 只用于诊断，不进入 LLM 输入或测试筛选。
- 已为 `export_neutral_tokens.py` 增加可选 `--span-policy full`（默认 noun 保持旧产物可复现），本地语法检查通过。等当前 LoRA 占用的 GPU 释放后，用相同抽样比较新旧 selector 的定位质量；若明显改善再考虑重导出。

### 2026-09-20 / 27 — v3 完整训练与评测结束

- v3 两轮训练均正常结束：373 条按题型抽样验证集 loss 第 1/2 轮为 0.0776/0.0771，最佳 checkpoint 为第 2 轮。训练后自动评测脚本也已正常完成，无后台训练进程残留。最佳 `best.pt`、完整结果、日志已复制到本地 `qa_pipeline/artifacts/qa_v3/runs/`。
- 完整 val 5,153 条：准确率 0.4939，多数类 0.4153；完整 test 4,219 条：准确率 0.5056，多数类 0.4646。按场景平均，test 0.5343（多数类场景平均 0.5031）。相比 v2 略有进步，但不同题型难度差异很大。
- test 按题型：近远 0.8065（多数类 0.6017），自车左右 0.5777（0.5498），自车联合方位 0.4568（0.4323），双目标相对位置仅 0.1628（0.1826）；最近两候选 0.5392（0.5588）。不能用总体 0.5056 掩盖双目标弱项。
- 同一个最佳 checkpoint 的 373 条固定 val 子集正常 token 准确率 0.5147，打乱同类别 token 后 0.4799；85/373 条预测类别改变。v3 比 v2 更依赖 token，但依赖仍有限。完整分析结果保存 `analysis_val.json`、`analysis_test.json`。
- 远程 GPU 释放后，已同步带 `--span-policy full` 的导出脚本并启动 128+128 条相同种子样本的完整中性短语 span pilot，PID 34777，日志 `/root/autodl-tmp/3eed_data/qa_v2_fullspan_pilot/export.log`；下一步和原 noun-span bank 按 object_id 成对比较定位 IoU。

### 2026-09-20 / 28 — 完整描述选择器 pilot 没有改善

- 128 train + 128 val 同样本配对比较：类别名词 span → 完整中性短语 span 后，train IoU≥0.25 从 0.6016 降至 0.5938，val 从 0.5547 降至 0.5156；val 平均 IoU 从 0.3117 降至 0.2805。约 80% 的选中 query 改变，但总体没有改善，因此不把全量 token 替换为 full-span。
- 为进一步区分“模型没有产生合适候选框”与“候选框存在但语言排序选错”，给导出脚本增加仅离线使用的 `--diagnose-candidates`：对全部 256 个候选框与 GT 比 IoU，报告 oracle 最佳 IoU，**不改变实际选中的 token**。
- 已启动 32+32 样本的 noun-span oracle pilot，PID 35296，日志 `/root/autodl-tmp/3eed_data/qa_v2_oracle_pilot/export.log`。它是上限诊断，不能用 oracle query 训练/测试最终 QA。

### 2026-09-20 / 29 — 证实主要瓶颈在语言选 query

- noun-span oracle pilot 正常完成。相同 32 条 val：实际文字选中 query 的 IoU≥0.25 为 0.5312，256 个候选框的 GT-oracle 最佳为 0.9688；train 分别 0.5625/1.0000。两侧各有 14/32 条属于“当前选中失败但候选池中有达标框”，实际 query 与 oracle 最优 query 恰好相同仅 8/32。
- 这说明 3EED 的 decoder 候选框通常存在可定位目标，但**语言到 query 的排序**在去空间提示后经常选错；既不能说 decoder 288 维特征完全坏掉，也不能说当前 top1 grounded object token 已经可靠。oracle 只用于离线诊断，不进入 QA 训练/测试。
- 为确定是否可改为每目标 top-k 隐式 token，又给诊断增加文字排序前 5/10 个候选的最佳 IoU 与 oracle 查询排名；已启动相同 32+32 pilot，PID 35430，日志 `/root/autodl-tmp/3eed_data/qa_v2_topk_pilot/export.log`。

### 2026-09-20 / 30 — top-k 候选诊断完成，形成 token 结论

- 相同 32+32 样本 top-k pilot 正常完成。val IoU≥0.25：文字 top1 0.5312、top5 中 GT 最佳 0.8125、top10 中 GT 最佳 0.8125、全部 256 候选 GT oracle 0.9688；train 相应为 0.5625/0.8750/0.8750/1.0000。正确候选经常落在 top5，但“top5 中 GT 最佳”仍只是离线上限，不是当前模型自动选中的结果。
- 新增 `TOKEN_AUDIT.md`，汇总 token 层位/维度、全量定位质量、候选池与排序诊断、v2/v3 token 打乱对照及后续可行改进。三个 pilot 的统计与逐目标记录已复制到本地 `qa_pipeline/artifacts/diagnostics/`，远程原产物保留。
- 当前结论：最后一层 288 维 query token 的来源正确、含单目标空间信息；但使用文字排名 top1 作为“grounded object token”**不可靠**，尤其同帧双/三目标。不能再靠单纯增加 QA 行数宣称解决此瓶颈；需要在不向 LLM 输入显式几何的前提下训练无 GT 推理时的 query 选择/聚合机制，并在完整数据复核 top-k pilot。

### 2026-09-20 / 26 — v3 完整评测与打乱对照完成：结论同 v2，负结果已备份

- 训练 2 epochs、2,226 步结束（val_loss 0.0776→0.0771），`best.pt` 为第 2 epoch（按 val_loss 选）。posttrain 脚本按计划自动完成两项评测：完整 val 5,153 条准确率 **0.4939**、test 4,219 条 **0.5056**，可解析率均 1.0。val 按题型：ego_near_far 0.8277 最高，which_is_left 0.6048、ego_side 0.5539、closer_of_two 0.5333、closest_of_three 0.5217(n=23)、closer_to_ego 0.5139、ego_location 0.4662，relative_location 0.1288 最低；test 同结构（relative_location 0.1628，closest_of_three 0.3333(n=27)）。训练集多数类基线 val 0.4153 / test 0.4646，总分只略高于答案先验。
- 打乱对照（同一 `best.pt`、每题型最多 50 条的固定 373 条 val 子集、同类别打乱 token）：打乱后 0.4799（179/373）。正常 token 的同子集成绩因**生成批次组成**不同有两个口径：按训练进程同款每题型分组顺序为 0.5147（192/373，差 3.5pp，85/373 条预测改变）；按评测脚本全 val 文件顺序切出为 0.4960（185/373，差 1.6pp，95 条改变）。两个口径都接近该子集"按题型多数类固定作答"的 0.4853，token 带来的净收益不超过 4 个百分点。
- 复现说明：同检查点、同 373 条、同贪心解码，仅生成批次顺序不同即可改变约 7 条结果（0.5147 vs 0.4960）。已用 `runs/quickval_repro/` 复现训练进程数字（完全一致），两个口径均保留；后续跨顺序比较 v3 数字时按 ±2pp 波动看待。
- 结论：18,274 条、8 种题型后模型仍主要依赖答案先验，0.49~0.51 不能当空间推理能力；relative_location（val 0.13 / test 0.16，子集 0.08~0.14 低于其多数类 0.26）是最大缺口。记录 17 提出的下一轮方向（题型/答案平衡、给 token 侧加有监督空间目标）仍属后续候选，不擅自开工。
- 已备份本地 `qa_pipeline/artifacts/qa_v3/runs/`：`full_eval/{eval_val,eval_test}.json`、`shuffle_probe/eval_val.json`、`quickval_repro/eval_val.json`、`full_lora/{train.log,history.json,best.pt}`（md5 `209edad8…` 与远端一致）、`posttrain.log`；第一轮 `best_epoch1.pt` 保留不动。备份后本地用 Python 重算全部数字，与远端一致。

### 2026-09-20 / 27 — 为什么模型不读 token：结合两线数据的机制解释（假说；未做新实验）

- 实测规律一：模型成绩与该信息的"浅层可读性"同序。val 对照——ego_near_far 线性探针 0.8937 → 模型 0.8277（其对照子集先验即有 0.68，净增益约 10 点）；ego_side 0.6942 → 0.5539；ego_location 0.5284 → 0.4662；relative_location 0.3009 → 0.1288（低于子集多数类 0.26，属"背错模板"而非"算反"）。单目标属性 token 里有浅层信号 → 模型学到一点；双目标关系连线性探针都读不出 → 模型基本躺平。
- 实测规律二：低 loss 是假象。答案约 7~9 个词中只有 1~2 个判断词，其余是固定模板；val loss 0.077 ≈（判断词 0.62 + 7×模板词 0.008）/8，反推判断词平均概率仅 ≈0.55（抛硬币偏一点），与生成 acc≈0.51 自洽；train loss 0.065 同理。即模板词把 loss 占满，指向 token 的梯度从一开始就很小——训练前 200 步 0.876→0.077 是"背模板"完成的时点，之后 token 通路几乎没被继续训练。
- 解释（推断）：三因素叠加——(1) 目标函数存在"近路"：题型是固定模板、部分答案分布偏斜，背模板即可得约半数分，优化先走近路并快速饱和；(2) 剩余任务要求"解码随机投影(288→3584) + 两个 token 间关系运算"，而 A 线已证该 LLM 连显式双坐标的减法都不做（D vs relgeo），B 线难度更高；(3) 没有任何监督逼它学这条通路（无答案平衡、无 token 侧辅助损失、无打乱/对比训练）。对照子集里恰好 50:50 的题型模型也只有 ~0.50，进一步说明不只是"背偏斜先验"，而是 token→判断的通路根本没建起来。该模式与多模态训练中已知的"单通道偷懒/捷径学习"一致，不是本项目特有 bug，修法也对应文献常规手段。
- 可验证的后续（均未开工，待用户/师兄确认）：(a) 零成本核查：逐题型 train/val 答案分布与"先验上限"统计；(b) 堵近路：答案逐题型平衡后重训；(c) 给 token 侧加监督（预测空间属性/关系的辅助损失，参考 probe_v3_tokens 的做法）；(d) 或按 A 线 relgeo 思路把关系显式算好再喂 LLM。

### 2026-09-21 / 28 — 与 A 线 C（exact 0.655）为什么差距大：口径与 token 对比说明（非新实验）

- 疑问："C 只给 token 有 0.655，B 线只给 token 只有 ~0.49"。核对两边口径确认不可直接比：
  - 题目与 exact 定义不同：A 线 C 只有"A 相对 B 在哪"一种问法，答案空间 2×2，且 `exact_accuracy` 只要求左右+前后两轴都对、**不含距离**（qa_eval.py 定义；实测 左右 0.939 × 前后 0.703 ≈ 0.655 自洽）。C 的距离单独计分：MAE 3.66m、Pearson 0.47，同样很差。B 线 8 题型、exact 为整句匹配，含 8 选 1 的相对位置题。
  - token 口径不同（关键）：C 用**原始人工 caption**（原文含 "slightly to the right of the center of the view" 等方位描述）导出的旧 token，定位准（预测中心中位误差 0.20m），方位信息大概率随 caption 进入 token；B 线为**删除方位词后重新导出**的中性 token（记录 03/06：移除位置提示后 Grounding 明显变难，IoU 0.38~0.57）。这正对应 B 线立项原因（记录 02：旧版完整 caption 的空间词可能进入 token）。
  - 同类题横比：同为"物体对物体关系"题，C exact 0.655 / 左右 0.939 vs B 线 relative_location 0.129。
- 解释（推断，可验证）：两条线合起来=该后端是"阅读器"而非"计算器"——信息在 token 里就会被读取（A 线方位近满分），需要"算"的（距离）或信息被抽走的（B 线关系）就失败。验证方式：两种 token 交叉跑对方题目（各需重训一轮，小时级）或先做旧 token 关系类线性探针与中性版对比。均未开工，待确认。
### 2026-09-20 / 31 — 明确师兄负责第一阶段 Grounding 训练与架构交接

- 根据用户最新分工，师兄自行训练第一阶段 Grounding；当前公开 3EED checkpoint 仅作为可运行的临时基线，不能称为师兄模型。
- 核对 `3EED/models/bdetr.py`、`point_backbone_module_v2.py`、当前导出脚本和远程 checkpoint 配置，形成 `qa_pipeline/GROUNDING_QA_ARCHITECTURE.md`。明确当前架构是 PointNet++ 风格点云 backbone → RoBERTa 与三层跨模态 encoder → 256 个候选 query → 六层 decoder → 每 query 一个 288 维 `last_query_features`；64 维 contrastive projection 只用于文字候选排序。
- 交接重点：师兄提供模型 checkpoint、结构/维度、候选 token 与 scene/object/expression ID 映射、推理期无 GT 的候选选择办法及独立 Grounding 评测。当前 top1 定位质量不足，尤其同帧双/三目标；QA 后端替换 token bank 后须重训 projector/候选聚合器并复测 LoRA。
- 严守隐式输入约束：GT box/center/距离仅用于离线 QA 标签、选择器监督和评测，不进入 LLM；本次只整理架构与接口，没有启动或修改训练。
### 2026-09-20 / 32 — 梳理位置输入与数据集设计

- 用户提出当前不能完全依赖未知的隐式 token，需决定怎样加入位置信息；本次复核 `3EED/models/bdetr.py`、当前 token 导出脚本、v3 token 探针及 v1 预测位置消融。
- 已知：公开模型 decoder 在 `loc_learned` 设置下以候选中心与尺寸生成位置嵌入；导出的 288 维 query 经过此交互，但没有可直接解释的 xyz 维度。v3 val 探针单目标远近 89.37%、双目标相对位置 30.09%；v1 test 仅 token/加预测中心/加预测相对几何的联合关系正确率为 59.94%/68.37%/79.82%，最后一项是预计算关系诊断上限。
- 新增 `qa_pipeline/POSITION_INPUT_DATASET_PLAN.md`：GT 几何与 QA 标签、版本化 Grounding 预测、LLM 输入适配器分层；推荐以预测中心经过位置编码与 query token 融合为主线，保留仅 token 基线与预测相对几何诊断。明确这种编码位置虽不在文本中显示数值，实质上仍是模型位置输入，不得称为纯隐式或使用 GT。
- 尚未修改现有 QA 数据、训练脚本或启动训练。下一步先在完整 val 审计师兄模型的候选选择与空间信息，再用同一 QA 拆分比较输入方案。
### 2026-09-20 / 33 — 开始复现 3EED 论文的 Grounding 基线

- 更正此前误解：师兄提出的是复现训练 **3EED 论文公开 Grounding 模型**，并非训练他的自有论文模型。本次目标是从头训练 Waymo/Vehicle 基线、独立验证、确认 token。
- 远程单张 RTX 4090（24 GB）空闲，3EED 数据、可用 `agiclass` 环境 PyTorch 2.1.0+cu121 和公开 checkpoint 已在位。核对官方脚本固定 GPU 4/batch 24，不适用这台机器；建立 `qa_pipeline/run_3eed_waymo_repro.sh`，适配 GPU 0/batch 4，其他主要模型/损失配置沿用官方。独立详细记录在 `qa_pipeline/GROUNDING_REPRO_LOG.md`。
- 确认既有远程代码 diff（忽略行尾）仅涉及 eval token 导出，训练 forward 默认路径未改。已启动 PID 37701 的一轮 `--debug` 短训练，日志 `/root/autodl-tmp/3eed_data/grounding_repro/smoke.log`；待确认 forward/backward、保存和 val 后启动正式训练。尚无新训练结果。
- 追踪更正：第一次 smoke 因误用 base Python 导致 `transformers` 缺失，在导入时停止；切换 `agiclass` 后 11 batch 训练、保存和 val 全部跑通，无 OOM/NaN。现已启动 **3EED Waymo Grounding 从头正式训练**，PID 38466，100 epochs、batch 4，日志 `/root/autodl-tmp/3eed_data/grounding_repro/full.log`；尚未完成，不报告为复现成功。
- 独立复核已有公开 checkpoint 在原始完整 Waymo val 3,782 条上的 Acc@0.25=0.8067、Acc@0.5=0.4989、mIoU=0.4674。抽查其导出文件确实是 288 维最终 query，与预测中心/尺寸共享 query_index；官方导出使用标注的正文本 span 作选择，不能称为无需标注的部署选择器。
- 正式训练第 1 轮 920 step 正常完成，用时 226.80 秒；已部署训练结束后的完整 Waymo val 评测与 token 导出监视任务 PID 39408，日志 `grounding_repro/posttrain.log`。已更正架构交接文档，把本次任务明确为复现 3EED 公开模型。新训练的 checkpoint 与指标尚未产生，后续以详细记录文件继续追踪。
### 2026-09-20 / 34 — 三个 Grounding 问题逐项核对

- 对照 `bdetr.py`、`encoder_decoder_layers.py`、`joint_det_dataset.py`、`grounding_evaluator.py`、`qa_token_export.py` 和探针脚本，新增 `qa_pipeline/THREE_GROUNDING_QUESTIONS.md`。
- 明确 `last_query_features [B,256,288]` 是最后 decoder 层输出、预测头之前的候选表示；64 维 `last_proj_queries` 仅用于与词投影匹配；同一 query_index 对应 token 与预测框。
- 更正“标注文本片段”的具体来源：Waymo 单目标数据加载器使用 GT 类别/同义词在 caption 中找首个类别词，形成 `positive_map`。官方 contrastive 选择器汇总这些词的相似度概率取 top1；完整描述仍参与 decoder 交互。该 span 来自数据集规则，不等同于未知用户表达中的自动指代解析。
- 位置参与 decoder 的代码路径已确认，但 token 内可解码的精确位置仍未知。现有单/双目标探针包含候选选择错误的影响，不能把 30.09% 双目标结果断言为“正确 token 本身完全不含位置”。后续需固定正确候选做上限探针，并与实际 top1 对照。
### 2026-09-21 / 35 — 温和停止 3EED 从头复现训练（转向隐式 QA 对照实验）

- 按新任务指令停止 3EED 从头复现训练及其训练后监视任务，**未清理任何产物**。
- 停止时间：2026-09-21 00:41:43（北京时间）= 2026-09-20T16:41:43Z。先对 torchrun（PID 38467）发 SIGINT：torchrun 抛出 SignalException 并终止训练 worker（PID 38533），约 20 秒内退出，外层脚本（PID 38466）随之退出；随后对监视任务（PID 39408，run_3eed_posttrain.sh）发 SIGTERM。核对后相关进程全部消失，GPU 显存 1 MiB、利用率 0%。
- 进度：最后完成**训练到第 30 轮**（ckpt_epoch_30.pth 于 00:35 保存）；第 30 轮验证进行到 63%（600/946）时中断，未完成。最后完成的验证是 epoch 25（00:16:23）：**Acc@0.25 0.7700 / Acc@0.5 0.4326 / mIoU 0.4263**（epoch 5/10/15/20 依次为 0.4868/0.7118/0.7427/0.7586，仍在上升、未收敛）。
- 产物路径（保留）：运行根 `/root/autodl-tmp/3eed_data/grounding_repro/`；checkpoint 在 `Train_waymo_Val_waymo/full_b4_seed0/ckpt_epoch_{10,20,30}.pth`（各 738,266,666 B）；日志 `full.log`（2.67 MB）、`Train_waymo_Val_waymo/full_b4_seed0/log.txt`、`smoke.log`、`posttrain.log`（停在 "Waiting for 3EED training PID 38466"）；脚本 `/root/autodl-tmp/3eed_data/run_3eed_waymo_repro.sh`、`run_3eed_posttrain.sh`；另有 `code_backup/`、`config.json`、`tensorboard/`、`predictions/`。
- 因训练未正常结束，`ckpt_epoch_last.pth` 未生成，监视任务的 `final_waymo_tokens/` 导出从未执行（目录不存在）。**这条"从头复现"训练线到此为止，不再作为 QA 前端。**
- 后续按新指令改用**公开 3EED checkpoint**（Grounding 冻结），QA 视觉输入只用模型学出的目标 query 与点云场景特征。下一步：核对导出接口（候选统一编号、多层 query、场景特征），小样本核对后再全量导出。
### 2026-09-21 / 36 — 隐式 QA 对照实验：富特征导出接口核对、脚本与小样本验证

- **接口核对完成**（`models/bdetr.py` forward）：候选=256 个 decoder query 槽位，编号 0..255；**6 层 decoder 原地精炼同一个 query 张量**（`query = self.decoder[i](query, ...)`，无重排），所以第 j 槽在各层就是同一候选（结构保证；脚本另断言最终层特征与旧 `last_query_features` 完全相等）。每层已有各自的预测（`{i}head_center/pred_size`，最终层 `last_center/...`）与 64 维对比投影 `{prefix}proj_queries`；文本侧 `proj_tokens`。场景特征：`fp2_features`（纯点云 backbone，跨模态编码前）与 `seed_features`（编码后、含文本交互），均 [B,288,N]。
- **改动**：`patch_bdetr_layers.py`（已同步远端并执行，备份 `models/bdetr.py.bak_rich_export`）——`return_query_features` 时保存所有层 `decoder_{i}_query_features`，最终层保留原 `last_query_features` 键，默认训练路径不变。
- **新脚本** `qa_pipeline/export_rich_features.py`（本地+远端同版）：对旧隐式 token 库的每个（场景, 中性描述）导出——6 层各自 256 候选的排序分数（full=整句中性描述、noun=首个物体名词，**均无 GT**）、top5 候选的 288 维特征/索引/分数、全部 256 候选的逐层 IoU（GT 框仅在预测后做离线诊断）、场景特征 mean/max（fp2 与 seed 各 2 个）、`records.jsonl` 按 `object_id` 对齐旧库 + `export_stats.json`。输出 `rank00/batch%06d.npz`。
- **小样本核对（每 split 16 条，共 32 条）全部通过**：npz 写出后可重读、形状/有限性断言通过；records 与旧库 `bank_index`/`neutral_query` 32/32 对齐；top5 分数与全分数向量自洽 32/32；`candidate_ids=0..255`。名词策略 top1 索引与旧库记录在 **batch 8 下 32/32 完全一致**（batch 4 时只有 6/16、7/16）——差异来源是**批内文本 padding 不同改变 softmax 归一化**的已记录批次敏感性（与 v3 生成 ±2pp 同类）；**全量导出统一用 batch 8**（与旧库同配置）。
- **全量导出已启动**（2026-09-21，背景）：`cd /root/3eedqa/3EED && python qa_pipeline/export_rich_features.py --batch-size 8 --num-workers 4 --out-dir /root/autodl-tmp/3eed_data/qa_features_v1`，日志 `/root/autodl-tmp/3eed_data/qa_features_v1_export.log`。全量模式会断言 roster 与旧库 6,130 条完全相同。预计 ~300 MB、约 0.5~1 小时。
- 小样本统计仅供参考（前 16 条 val 恰是难子集）：oracle 候选 acc@0.25=1.0，说明 256 候选里总有可用目标。下一步：等导出完成核对全量 stats（名词策略应复现旧库 train acc@0.25 0.5728 / val 0.4750），再做 T0~T9 轻量筛选。
### 2026-09-21 / 37 — 轻量筛选框架（T0~T9）与 T0 基线

- 新增 `qa_pipeline/screen_token_groups.py`（本地+远端同版）。**这是筛选工具，其分数不构成大模型 QA 成绩。**
- 固定题集：qa_v3 `scenario_qa`（train 8,902 / val 5,153 / test 4,219，序列隔离拆分），答案键 13 个：{A,B,C}（which_is_left/closer_to_ego/closer_of_two/closest_of_three）、near/far、left/right（ego_side）、{left_front,right_front,front}（ego_location）、{left,right,left_front,right_front,left_behind,right_behind,front,behind}（relative_location）。
- 轻量模型（所有组完全相同，每组只换视觉输入）：冻结 RoBERTa-base 句向量（768，均值池化，不微调）+ 视觉块拼接 + 2 层 MLP（512、dropout 0.1、lr 1e-3、wd 1e-4、batch 256、40 epoch、seed 42、按 val 取最佳 epoch、按题型屏蔽答案空间）。视觉特征按 `object_id` 从富特征导出关联，标准化统计量只用 train 行。多目标按角色 A,B,C 顺序拼接各目标特征。
- 实现口径（固定，先记下以免以后混淆）：top-k 一律按**整句中性描述**的排序分数取；T3 的"中间层"=第 2 层（0..5 索引的中间偏左）；T4=最后两层 top1 的**拼接**；T5 的一个场景特征=seed_mean；T6 的多个=4 个全用；T7/T8/T9 的 top3/top5 均为最后一层。每组保存 `screening/runs/{T#}/`：`metrics.json`（配置、样本数、分题型、grounding 分组诊断）、`errors_{val,test}.jsonl`（错误样本）、`screen_model.pt`。
- **T0（纯文字）已完成**：val 0.4854 / test 0.4712。对照 v3 大模型（Qwen+LoRA，val 0.4939 / test 0.5056）——**纯文字轻量模型几乎追平大模型**，与该题集"主要靠题型先验"的既有结论一致，也再次说明 v3 的 0.49~0.50 里 token 贡献极小。T1~T9 必须显著超过 T0 才能说明视觉/隐式 token 通道起了作用。
- 待办：富特征全量导出完成（当时约 104/767 批）→ 核对全量 stats → 依次跑 T1~T9 → 汇总对照表并挑 2~3 组进 Qwen。
### 2026-09-21 / 38 — 富特征全量导出第二次：修文件覆盖缺陷后重跑

- **发现缺陷（第一次全量导出，已完成）**：`rank00/` 只有 386 个 npz（应为 767 = train 381 + val 386）。原因：文件名 `rank00/batch%06d.npz` 不含 split 前缀，而 `batch_idx` 在每个 split 内从 0 重新计数，导致 **val 的 npz 覆盖了 train 的同名文件**。抽查确认：train 侧 40 条记录里 39 条文件内容与记录不符，val 侧 40/40 正常。运行 1 的打分统计是推理时直接算的，**不受文件覆盖影响**（val：名词策略 acc@0.25 0.4750、iou_mean 0.2753，与旧库完全一致；query_index 与旧库 3082/3082 一致；各层 full top1 0.4737~0.4786、top5 recall 0.68、oracle 0.98）。
- **修复（`export_rich_features.py`，本地+远端）**：① 分片名加 split 前缀 `rank00/{split}_batch%06d.npz`；② 写出后立即重载校验（形状 `[batch_n,6,5,288]` + 首行名词 top1 索引必须一致），不符即抛错；③ 运行结束按 split 断言分片数 = ceil(eligible/batch_size)，并把实际数量写入 `export_stats.json` 的 `shards`。
- 第一次的损坏产物改名保留为 `/root/autodl-tmp/3eed_data/qa_features_v1_run1_corrupt/`（57 MB，确认第二次无误后删除）；运行 1 日志保留为 `qa_features_v1_export.log`。
- **第二次全量导出已启动**（2026-09-21 01:35 北京时间）：同参数（batch 8、num_workers 4、train+val），日志 `/root/autodl-tmp/3eed_data/qa_features_v1_export2.log`。启动时已打印 `[train] eligible 3048 roster identical_to_bank`（roster 与旧库 6,130 条完全一致）。待完成后核对：767 个分片、records 6,130、train+val 均无文件-记录不符、名词策略复现旧库 train 0.5728 / val 0.4750。
- 等待方式改为本地轮询远程进程（用 `[e]xport_rich_features.py` 括住首字母避免 pgrep 自匹配——上一轮监视脚本因自匹配永不退出，已清理）。
### 2026-09-21 / 39 — 富特征导出核验通过；筛选脚本 T1 崩溃修复并重跑

- **导出第二次（运行 2）全部核验通过**，损坏问题彻底解决：
  - 767 个分片（train 381 = ⌈3048/8⌉、val 386 = ⌈3082/8⌉）、records 6,130，文件名带 split 前缀。
  - 新增 `qa_pipeline/verify_rich_export.py` 逐条交叉核对，结果 **VERIFY PASS**：train 3048/3048、val 3082/3082 条记录与其 npz 实际内容（名词策略 top1 候选编号 + 该候选 IoU）完全一致，分片数与预期一致。
  - **名词策略完全复现旧 token 库**：train acc@0.25 0.5728 / iou_mean 0.3412 / query_index 与旧库 3048/3048 一致；val 0.4750 / 0.2753 / 3082/3082。roster 两个 split 都是 `identical_to_bank`。
  - 拒收统计：train {spatial_cue 551, noun_not_target_category 48, too_long 28, no_noun 7}（3682 → 3048）；val {spatial_cue 612, noun_not_target_category 60, too_short 4, too_long 22, no_noun 2}（3782 → 3082）。
  - 逐层诊断（full 策略 top1 / top5 recall / oracle，acc@0.25）：train layer_0 0.5784 / 0.7749 / 0.9944，layer_5 0.5732 / 0.7680 / 0.9856；val layer_0 0.4786 / 0.6934 / 0.9848，layer_5 0.4737 / 0.6823 / 0.9744。**同一候选在 6 层都有特征，层间差异很小**（top1 ±0.005），top5 召回 ~0.68~0.77、oracle ~0.98。
  - 目录 `/root/autodl-tmp/3eed_data/qa_features_v1`（106 MB）＝ rank00/ + records.jsonl + export_stats.json；运行 1 的损坏归档 `qa_features_v1_run1_corrupt/` 已删除（核验后）。
- **脚本缺陷与修复（导出）**：结束时断言报 `KeyError: 'eligible'`——第 414 行 `stats["splits"][split] = split_stats` 整体覆盖了该 split 条目，把先前写入的 loaded/eligible/roster_check/rejections 挤掉。改为 `.update()` 合并；future 重跑会正常写 stats。因本次断言在写 `export_stats.json` 之前崩溃，该文件由脚本自身打印在运行日志里的分片统计**原样转录**重建，文件内 `_provenance` 字段记录了此事。`verify_rich_export.py` 自身也有个小错（形状检查写成 `shape[2:]`），已改为 `shape[1:]`。
- **T1 首次运行崩溃修复**：`np.stack` 报"形状不一致"——`build_visual` 按每行实际目标数拼接特征，1/2/3 个目标使行向量长度 288/576/864 不等（T0 无视觉输入，掩盖了此问题）。数据核对：qa_v3 每题 1~3 个目标（12,301/5,298/675 条），角色恒为 A/B/C 连续。**修复**：固定 **3 个角色槽位**，逐个角色拼接、缺的角色补零；标准化改为**按角色块**、只用 train 行统计，缺失块在标准化后保持精确 0（避免"补零被标准化成非零大数"）。崩溃日志留档 `screening/screen_groups_attempt1_crash.log`。
- **T1 冒烟（2 epoch，输出到 runs_smoke/，不参与汇总）**：visual_dim 864 = 3×288，val 0.4473 / test 0.4432（只练 2 轮，低于 T0 属正常）。已见期望信号：**grounding 命中时 acc 0.6010 vs 未命中 0.3747**。
- **T1~T9 正式重跑已启动**（2026-09-21 02:2x 北京时间，日志 `screening/screen_groups.log`），完成后跑 candidate_recall 与汇总表。
### 2026-09-21 / 40 — T0~T9 轻量筛选完成（+3 个归因对照），候选召回统计

- 全部 10 组 + 归因对照 X1~X3 跑完，每组约 1 分钟。汇总表：`screening/comparison.md`（已复制到本地 `reports/screening/comparison.md`）；各组明细 `screening/runs/{组}/metrics.json`、`errors_{val,test}.jsonl`（最多 100 条错例）、`screen_model.pt`。**这些是轻量筛选分数，不是大模型 QA 成绩。**

| 组 | 视觉输入 | val | test | 关系题 val | 关系题 test | 命中时 val | 错定位时 val |
|---|---|---|---|---|---|---|---|
| T0 | 纯文字 | 0.4854 | 0.4712 | 0.3143 | 0.3260 | - | - |
| T1 | 末层 top1 token | 0.4789 | 0.4726 | 0.3324 | 0.3205 | 0.6258 | 0.4096 |
| T2 | 次末层 top1 | 0.4774 | 0.4567 | 0.3185 | 0.3292 | 0.5206 | 0.4570 |
| T3 | 中间层 top1 | 0.4772 | 0.4620 | 0.3324 | 0.3299 | 0.5744 | 0.4313 |
| T4 | 末两层 top1 | 0.4683 | 0.4641 | 0.3271 | 0.3323 | 0.5653 | 0.4224 |
| T5 | top1 + seed_mean | **0.6142** | **0.5880** | 0.4126 | 0.4028 | 0.7461 | 0.5519 |
| T6 | top1 + 4 场景特征 | **0.6315** | **0.5900** | **0.4591** | 0.4020 | 0.7545 | 0.5733 |
| T7 | 末层 top3 | 0.4570 | 0.4691 | 0.3399 | 0.3182 | 0.5937 | 0.3924 |
| T8 | 末层 top5 | 0.4685 | 0.4461 | 0.3522 | 0.3401 | 0.5526 | 0.4287 |
| T9 | top3 + 4 场景特征 | 0.6212 | 0.5900 | 0.4436 | **0.4318** | 0.7449 | 0.5627 |
| X1 | top1 + fp2_mean | 0.5445 | 0.5077 | 0.4249 | 0.3386 | 0.6651 | 0.4876 |
| X2 | 仅 seed_mean | 0.6187 | 0.5911 | 0.4126 | 0.4044 | 0.7854 | 0.5399 |
| X3 | 仅 4 场景特征（无 token） | 0.6315 | 0.5916 | 0.4543 | 0.4075 | 0.7594 | 0.5710 |

- **结论 1（token 单独无效）**：T1~T4、T7、T8 全部与 T0 持平或略低（val 0.457~0.479 vs 0.485），**只加目标 token 不提升总体正确率**；仅在定位命中样本上有明显优势（T1 命中 0.626 vs 错定位 0.410，说明 token 确实带空间信息，但错定位时反而拖累）。改变层（T2/T3）、合并两层（T4）、扩大 top3/top5（T7/T8）都不改变这个结论。
- **结论 2（成绩基本来自场景特征通道）**：T5/T6/T9 ≈ +13~15pp。归因对照：**X3（完全不给 token，只用 4 个场景特征）= T6（token+场景特征），val 完全相同 0.6315、test 0.6315 vs 0.5900** → 在最好的组里**目标 token 的边际贡献 ≈ 0**。X2（仅 seed_mean）= T5（0.6187 vs 0.6142）同样说明 token 无增益。X1（纯点云 fp2 特征 + token）0.5445 居中，说明纯点云几何本身也带相当多信息。
- **注意（写报告时必须说清）**：seed 特征是跨模态编码器的输出、以该目标的中性描述为条件（同场景不同目标取值不同），属指令允许的输入通道，但它是**场景级/上下文级**信息，与"目标 query token"是两条不同来源；不能把 T5/T6/T9 的提升记作"目标 token 有效"。fp2 特征是编码前纯点云特征。
- **候选召回**（`screening/candidate_recall.json`，末层整句策略，离线对照 GT）：val R@1 0.451 / R@3 0.597 / R@5 0.657（n=1590）；test 0.500 / 0.656 / 0.709（n=1414）；train 0.573 / 0.713 / 0.766（n=2957）。**层间差异很小**（各层 R@1 相差 <0.01），且名词策略与整句策略几乎相同。
- 供第 5 项选组的依据齐了：至少含 T1 基线；按 val 关系题成绩 + 候选召回，建议 **T1（基线）+ T6（val 最好）+ T9（test 关系题最好）**；可另加 X3 作"无 token"对照以直接检验大模型是否用到 token。
- 下一步：`train_group_qa.py`（已写好并通过导入与打乱映射自查）先做 Qwen 冒烟，再按组逐一开始 LoRA 训练。
### 2026-09-21 / 41 — Qwen 管线冒烟通过；修 projector 加载顺序缺陷；T1/T6/T9 正式运行已启动

- **冒烟（`qa_v3/runs/group_smoke/T1`，退出码 0，全部通过）**：T1 训练 16 条 1 轮 + 三趟评测（normal / `--swap` / `--shuffle-seed 42`）全跑通。产出 `best.pt`、`last.pt`、`history.json`、`eval_val.json`、`eval_val_swap.json`（含 swap 块：n_applicable 20、acc_normal/acc_swapped、flip_rate、content_rate、分题型）、`eval_val_shuffle42.json`。冒烟分数无意义（欠训练），只验证管线。**注意：冒烟只覆盖 T1（288 维，不触发 projector 重建）。**
- **发现并修复一个会在 T6/T9 评测阶段才炸的缺陷**（`train_group_qa.py`，本地+远端同版）：基类 `QATrainer.__init__` 先建 288 维 projector 并**立刻**加载 `--init-projector`，之后 `GroupTrainer` 才按组维数重建——宽维组（T6 1440、T9 2016）会出现：① 基类加载时维数不符（`_load_expanded_projector` 只能把当前层"加宽"，不能让 1440 权重塞进 288 层）直接报错；② 即使不报错，重建会把已加载的训练权重换成随机初始化，评测结果失效。**修复**：宽维组先临时置空 `args.init_projector` 跳过基类加载，重建 projector **之后**再显式 `load_checkpoint`。T1 路径不受影响（288 = 基类维度，行为与冒烟完全一致）；远端 `import` 自检通过。
- **新增 `--only-multi-ref`**（`train_group_qa.py`）：评测探针只跑引用 ≥2 个目标的题行（即双目标及以上关系题，val≈1684 / test≈1380，约全量 1/3）；结果 JSON 里记 `only_multi_ref` 字段。理由：swap/打乱探针对单目标题无意义，全量跑要多花一倍以上生成时间。**注意：宽维组的 projector 是随机初始化开始训练的**（与 T1 同样从零训，组间口径一致；不借用 v3 权重）。
- **新增 `run_group_qa.sh`**（本地+远端）：顺序跑 T1 → T6 → T9，独立目录 `qa_v3/runs/group_{组}/` 互不覆盖；每组四个阶段——训练（组内 val 监视沿用 v3 的每题型 50 条口径选最佳 epoch）→ 全量 val+test normal（主成绩）→ 双目标行 A/B 交换 → 双目标行打乱特征（seed 42）；阶段标记带 UTC 时间，失败即停并打 `FAILED`。
- **新增 `aggregate_group_runs.py`**：读各组 eval JSON 出三张表（主表：全量 val/test 正确率、关系题、按定位命中/未命中分组、解析率；swap 表；打乱表），已用冒烟 JSON 实测字段对齐。
- **正式运行已启动**：2026-09-21 02:47:08 北京时间（18:47:08Z），`bash run_group_qa.sh`（PID 60502，nohup），日志 `/root/autodl-tmp/3eed_data/qa_v3/group_runs.log`。T1 训练：train 8,902 / 组内 val 监视 373 条，每轮 1,113 步、共 2,226 步、warmup 66，projector in_dim 288；02:56 已到 step 300（loss 0.073，GPU 19.1 GB）。实测生成速度约 2 行/秒（含模型加载），单组评测约 2.5~3 小时、三组合计约 12 小时。
- **选组依据**（指令第 5 项：至少含 T1，其余按验证集双目标关系成绩与候选召回）：**T1（必备基线）+ T6（val 关系题最好 0.4591）+ T9（test 关系题最好 0.4318）**。另可选 X3（无 token 对照）作第 4 组，等这 3 组完成后再定。
- 监控：本地定时任务每 2 小时查一次（会话内有效，7 天后自动过期）；本地断线不影响远端 nohup。下一步：等运行完成 → `aggregate_group_runs.py` 出对照表 → 最佳组完整评测与负结果整理（指令第 6 项最终交付）。
### 2026-09-21 / 42 — T1 全流程完成；T6 评测进行中；口径提醒

- **T1（token-only）四阶段全部完成**（18:47Z 启动，19:51Z DONE，约 1 小时 4 分）。独立目录 `qa_v3/runs/group_T1/`：`best.pt`/`last.pt`（02:59 保存最佳，03:10 收尾）、`history.json`（2 epoch，每轮约 680 s，val_loss 0.0775→0.0776）、`eval_{val,test}{,_swap,_shuffle42}.json` 六个评测文件齐全。**主成绩（全量）：val 0.4467（n=5153）/ test 0.4750（n=4219）**；组内监视子集（每题型 50 条，n=373/377，与 v3 同口径）：val 0.4987 / test 0.4430，关系题 0.4484 / 0.4053。定位命中时 val 0.6191 vs 未命中 0.3653（与轻量筛选同向）。
- **重要口径提醒（写最终报告必须说清）**：v3 当年的 val 0.4939 / test 0.5056 是**每题型 50 条子集**上的分数，本次主成绩是**全量拆分**；同一模型两种口径可差 5pp（T1：子集 val 0.4987 vs 全量 0.4467；差异来自场景构成——全量里相对位置题占比近半且更难）。**不要直接拿全量分数和 v3 子集分数比较**。
- **T6（token+4 场景特征，1440 维）**：训练完成（19:51→20:15Z，val_loss 同量级）；**修复后的加载顺序在生产中生效**——日志顺序为 `projector rebuilt | in_dim 1440 -> 1024 -> 3584` → `loaded checkpoint ... | projector yes | lora yes`，即先重建后加载，权重正确。子集 val 0.4987 / test 0.4536，关系题 0.4574 / 0.4361。全量：**val 0.4485 / test 0.4767**，关系题 0.3303 / 0.3683。**全量关系题上 T6（0.3303）与 T1（0.3330）没有差别**——轻量筛选里场景特征带来的 +13pp 关系题增益**没有迁移到大模型**，反而 T6 全量非关系题、子集 test 都略低于 T1。待 T6/T9 全部完成后统一解读。
- **探针口径缺陷（写报告时必须注明）**：`mirror_key` 只对含 A/B 字母的答案键（左右/近远二选、三选一）有效；相对位置题等键（left/right/front/behind…）不含字母，镜像=恒等，此时 `flip_rate` 退化为"预测是否不变率"、`content_rate` 退化为准确率。T1 val swap：flip 0.497 / content 0.286（含全部题型混合）；相对位置题单独看是预测完全不变（flip 1.0）。**初步迹象：交换 A/B 特征块后大模型答案基本不变**（即答案主要来自文字/标签顺序，不是特征内容），与轻量筛选"token 边际贡献≈0"一致。
- T1 val 打乱特征（同 (split, category) 供体，seed 42）：子集行 n=1871，acc 0.3405，与未打乱 0.3303（同 1871 行）**基本持平**（test：0.3472 vs 0.3668，低约 2pp）；即把特征换成别的物体的特征，成绩不掉。
- 进度：T6 的 swap 阶段 test 生成中（20:44Z），随后 shuffle 阶段，预计 21:0xZ 完成并自动进入 T9。定时检查继续，无需人工干预。
### 2026-09-21 / 43 — T1/T6/T9 全部完成（0 失败）；三组主成绩无差别；最终报告与对照表已出

- 全流程结束：`GROUP_RUNS_ALL_DONE` 于 2026-09-21 06:02:54（北京时间）= 2026-09-20T22:02:54Z；日志全文 0 处 FAILED。T1（18:47Z–19:51Z）、T6（19:51Z–21:0xZ）、T9（20:55Z–22:02Z）三组各自目录六件套齐全（`best.pt`/`last.pt`/`history.json`/`eval_{val,test}{,_swap,_shuffle42}.json`）。三组都选中第 1 轮为最佳（val_loss 组间差异 <0.001）。
- **主成绩（全量，test 为主）**：T1 val 0.4467 / test **0.4750**；T6 0.4485 / **0.4767**；T9 0.4475 / **0.4736**。关系题：T1 0.3330/0.3707、T6 0.3303/0.3683、T9 0.3324/0.3566。**三组一致，换视觉输入组别没有可测影响。**
- **探针**：A/B 交换后 acc 不变（0.3303 vs 0.3303 等）；含 A/B 字母的题型 `flip`≈0.02–0.18（完全内容驱动应≈1）→ 答案不随特征块交换改变；`relative_location`（非字母键，flip=不变率）0.77–0.81，即 19–23% 的题预测会变，是唯一有特征依赖迹象的题型。打乱特征（同 (split, category) 供体）成绩变化 ±1.5pp 内、无下降。
- **分题型**：ego_near_far 最高（0.67–0.68）；**relative_location 三组都 ≈0.19（八选一，随机 0.125）**，接近随机。
- **负结果（已写入报告）**：① 轻量筛选里场景特征的 +13~15pp 增益未迁移到大模型；② token 通道在大模型里测不到贡献（T1≈T6≈T9，与 X3==T6 的筛选结论一致）；③ relative_location 没学会；④ 定位命中行分数高（0.62–0.63 vs 0.36–0.40）只是行难度相关性，结合探针不能当作"模型在用特征"的证据。
- **定稿产物**：本地 `reports/group_qa_final_report.md`（设置、主成绩、探针、分题型、口径对照、负结果、限制、下一步）+ `reports/group_comparison.md`（远端 `qa_v3/group_comparison.md` 的副本，由 `aggregate_group_runs.py` 生成）。
- **限制（报告第 7 节）**：缺大模型层面"完全无视觉输入"对照（T0-LLM）；单种子 2 epoch；子集口径样本小、与全量不可比；探针指标对非字母答案键退化；宽维 projector 从零训、场景特征未标准化。
- 保持现场：三组 checkpoint 与全部评测 JSON 保留在远端 `qa_v3/runs/group_*`，未清理。定时检查任务已按流程删除。
- **决定（2026-09-21 用户确认）：本阶段到此定稿，不追加"大模型 T0-LLM 对照"**。`reports/group_qa_final_report.md` 标记为定稿，T0-LLM 作为未做实验列在第 8 节；若以后需要，证据链与现场都齐备，可随时补跑。
### 2026-09-21 / 44 — 复核最新实验并更正最终报告的评测口径

- 按用户要求复读本地最新 `WORK_LOG.md`、筛选表、`group_qa_final_report.md`，并到远程核对 Grounding 与 T1/T6/T9 产物。远程 `GROUP_RUNS_ALL_DONE`，三组全量 val/test JSON 都存在；GPU 空闲。
- 确认 3EED 从头训练按 35 条记录于第 30 轮验证中被温和停止，不是自发故障或训练完成。最后**完整**验证是 epoch 25：Acc@0.25=0.7700、Acc@0.5=0.4326；epoch 30 checkpoint 保留但其 val 未完成。公开 checkpoint 的原始 Waymo val 参照为 0.8067/0.4989。
- 独立读取旧 v3 `full_eval/eval_val.json` 和 `eval_test.json`，确认旧版 0.4939/0.5056 是 **n=5,153/4,219 全量**，不是原报告误写的 n=373/377 子集。旧监视子集 `full_lora/eval_val.json` 才是 n=373、0.5147。更正 `reports/group_qa_final_report.md` 第 2、5 节的口径与 test 组间极差（0.31 个百分点，不是 3.1）。历史日志 42 条所写“旧 v3 是子集”被本条更正；原始实验文件未改。
- 远程复核 T1/T6/T9 全量 test 0.4750/0.4767/0.4736，样本数均 4,219；相对位置题 test 0.1875/0.1908/0.1941（n=608）。组别增加场景特征/top3 未观察到稳定总体收益；打乱/交换特征的原报告限制仍有效。未新增训练。
### 2026-09-21 / 45 — 核对单目标与联合多目标 Grounding 的关系

- 根据用户提出的问题，核对 3EED 论文、`scripts_multi/`、`Joint3DDataset`、Hungarian loss、evaluator 和当前 `qa_v3/scenario_qa/qa.jsonl`。新增 `qa_pipeline/MULTI_OBJECT_GROUNDING_DECISION.md`。
- 当前多物体 QA 是同一场景中 A/B/C 各自一条描述、分别运行单目标 Grounding，再将各自 token 一起送给 QA；因此**不必须**先改网络为一句话联合预测。风险是多个独立预测可能错选或选到同一物体；v3 val 双目标全部命中 IoU≥0.25 仅约 9.5%。
- 3EED 论文本身支持单句多目标任务；仓库有多目标 loader、训练脚本、Hungarian loss。但当前远程缺 `waymo_multi_{train,val}_info.pkl`，contrastive evaluator 有 `assert num_obj == 1`，现有 QA token 导出也只支持单目标，`val_multi_3eed.sh` 的 checkpoint 参数为空。故联合多目标训练/评测需要补齐工作，不能把已有单目标实验说成联合多目标成绩。本次仅审计并记录，未重启已停止的训练。

### 2026-09-21 / 46 — 启动联合多目标 Grounding 主线，数据与评测补齐

- 用户决定以一次输入整句、同时定位 A/B/C 的联合多目标模型为主线；旧单目标逐个调用保留作比较基线。`MULTI_OBJECT_GROUNDING_DECISION.md` 已更新。
- 新增 `qa_pipeline/build_multi_grounding.py`，从 neutral bank 的 GT object/box 生成自建 2–3 目标标注；每目标 `target_spans` 对齐对应框，同一 sequence 只进一个 split。生成 `qa_pipeline/artifacts/multi_grounding/waymo_multi_{train,val,test}_info.pkl`：train 908（604 场景）、val 614（320）、test 397（295），三划分序列交集均为 0。数据是自建启动集，不是官方 multi 公开标注。
- 修复 `3EED/src/joint_det_dataset.py` 多目标 loader：读取有序目标文字片段、允许缺失无用的 2D box、验证每个框都有非空 positive map，并支持独立 test split。修复 `3EED/src/grounding_evaluator.py` 多目标路径：Top-1 query 一对一分配、逐目标 IoU、所有目标全命中率；单目标路径保持原实现。新增 `main_utils.py` 的模型权重初始化入口及 `scripts_multi/train_multi_bootstrap.sh`。
- 已将代码、三个 pkl 同步到远程 `/root/3eedqa/3EED/`。远程 4090 用 `python -m scripts_multi.smoke_multi_dataset` 通过全部 908/614/397 条标注的文本框对齐和三 split 的真实点云取样；本地 py_compile 通过。尚未完成端到端训练与 token 导出。
- 08:37 北京时间在远程启动 `/root/3eedqa/3EED/scripts_multi/train_multi_bootstrap.sh`，公开单目标 `ckpt_6384.pth` 仅初始化模型，新的优化器训练 10 epoch（batch 4、lr 2e-5、每 2 epoch 验证/存盘）。训练日志 `/root/autodl-tmp/3eed_data/multi_grounding/train.log`，run 目录 `/root/autodl-tmp/3eed_data/multi_grounding/bootstrap_run/Train_waymo-multi_Val_waymo-multi/0921_0837/`，后台父 PID 84462。epoch1 用时 61.71 秒，epoch2 用时 56.42 秒；验证正在进行。
- `utils/qa_token_export.py` 已在本地扩展为多目标有序 token 导出：每角色通过标注文字片段评分，匈牙利分配不同 query，保存每目标 288 维特征、对应预测框、query 索引、target_count；单目标输出维度兼容旧格式。该导出仍依赖提供目标文字片段，部署时需从问题解析片段；尚未远端同步/实测。
- 已同步并远程 CPU 冒烟测试多目标 token 导出，合成双目标样本得到不同 query `[0,1]` 与 `(1,2,288)` 特征；补充独立 `--eval_split test`、测试评估脚本及 selected/oracle 摘要脚本。当前训练第二轮验证 all-target Acc@0.25=0.0277、@0.5=0.0212，第四轮 0.0244/0.0163，仍很差。旧单目标分别调用在**同一自建目标组合**上按已有 neutral bank IoU 计算的 all-target 基线：val 0.0798/0.0342，test 0.1461/0.0529；联合模型此时不能替换旧方案。
- 10 epoch 完成；验证 Acc@0.25：ep2 0.0277、ep4 0.0244、ep6 **0.0358（最佳）**、ep8 0.0293、ep10 0.0261，按 val 选 ep6；最终重复 eval 的随机采样有约 0.3pp 波动。独立 test 397 条：contrastive unique Top-1 all-target Acc@0.25 **0.0353**、@0.5 **0.0277**；逐目标 0.0941/0.0784。用 GT 中心作**离线分析**，最近 16 个候选的逐目标命中 0.3559/0.2256、全目标命中 0.2821/0.1688（只是候选覆盖下界，不是模型无 GT 推理成绩）。测试导出 100 个 batch NPZ，两个或三个有序 288-D token/框/唯一 query；checkpoint ep6 在上述 run 目录。由于联合模型远低于旧逐目标基线，不进入 QA 主模型。
- 为区分文字打分头，增加多目标 soft-token 分支的整句全命中指标 `SoftAll@...`；正在用固定 ep6 在 val 上比较，之后决定是否另测 test。更新后的测试预测记录添加 caption 哈希及层名避免同帧多 caption 被覆盖。
- 固定 ep6 的独立 val 重评：contrastive all-target @0.25=0.0212，**soft-token=0.0651**；@0.5 分别 0.0081/0.0456。验证集选 soft-token 作为导出/主选择头（不根据 test 挑）。已改 `qa_token_export.py`，有序 token NPZ 现在包含 `object_ids_json`；远程 CPU 合成双目标导出冒烟通过。首次 test contrastive token 文件移到 `tokens_test_contrastive/` 保留；正在重评 test 并输出 soft-token 文件至 `tokens_test/`。
- 同一 ep6 test 重评完成：contrastive all-target @0.25=0.0353/@0.5=0.0277；**soft-token @0.25=0.0579/@0.5=0.0353**，高于 contrastive 但仍显著低于旧单目标逐个定位同组合基线 test 0.1461/0.0529。`scripts_multi/verify_multi_export.py` 检查全部 soft-token NPZ：100 文件、397 样本、829 目标，每行 `object_ids_json` 长度与目标数一致，query 一对一且有效 token 全 288 维。当前多目标模型是可运行的原型，不适合作为 QA 主链输入。下一步应先改联合目标的语言选框和训练数据质量，再谈替换。
- 新增可阅读报告 `qa_pipeline/MULTI_GROUNDING_EXPERIMENT_REPORT.md`，记录输入/输出、数据 split、训练参数、同组合对照、候选覆盖诊断、产物路径和下一轮实验。此轮没有替换旧 QA checkpoint，也没有修改 QA GT 标签。

### 2026-09-21 / 47 — 正确定位 token 的 QA 条件诊断

- 用户同意做“正常选出的 Grounding token 中，仅取所有目标定位正确的多目标题”诊断。明确口径：GT box 只离线判断是否进入诊断集，绝不用于从候选 query 中挑 token，也不作为 QA 模型输入；答案仍取原始 QA 的 GT 标签。三组使用完全相同的题目：旧单目标 token、联合多目标正确 token、同 split 同类别的其他物体 token 打乱。
- 新增 `qa_pipeline/build_correct_multi_qa.py`，按导出 NPZ 的 `object_ids_json` 与 v3 `object_refs` 精确集合匹配，用原 3EED rotated IoU 函数逐目标核对阈值 0.25。测试集 397 个联合 Grounding 样本里 23 个全部目标命中；原 v3 有 1,276 道多目标题，其中 89 道对应这 23 个正确组合，且全部可以找到同类别不同物体的 shuffle 供体。生成 `/root/autodl-tmp/3eed_data/multi_grounding/qa_correct_only/{original,joint,shuffled}/` 三套同 ID QA 输入、私有 GT 与隐式 token bank；联合 bank 48 个角色 token。89/1276 的题目保留率约 6.97%，不能把筛后 QA 准确率当端到端成绩。
- 新增 `qa_pipeline/run_correct_multi_qa.py`，加载已有 v3 `full_lora/best.pt` 一次，冻结模型依次生成三组回答，产出逐题配对结果。目前远程 `/root/autodl-tmp/3eed_data/multi_grounding/qa_correct_only/eval_frozen_v3.log` 正在运行；输出目录 `eval_frozen_v3/`。没有重新训练，也没有改原 QA 题目/答案。
- 三组推理完成，projector 和 LoRA 都实际加载；同 89 道题旧 token 43/89=0.4831、联合正确 token 44/89=0.4944、联合 token 打乱 45/89=0.5056。联合 vs 打乱只有 12/89 道生成文本变化，其中打乱让 2 道由对变错、3 道由错变对；没有观察到正确 token 的可靠收益。相对位置题仅 42 道，联合 10/42、打乱 11/42。冻结模型原本在旧 token 上训练，联合 token 分布变化与 23 个成功组合的样本量限制了解释。
- 从远程复制构建摘要、评测摘要及三套 89 行逐题 JSON 到本地 `qa_pipeline/artifacts/multi_grounding/qa_correct_only/`，新增 `qa_pipeline/CORRECT_TOKEN_QA_DIAGNOSTIC.md` 写明条件筛选、配对结果和不能把筛后成绩当端到端成绩的限制。远程 GPU 已空闲；原 checkpoint 和 QA GT 未改。

### 2026-09-21 / 48 — 下载 3EED 原始数据小样本到本机

- 按用户要求，从远程原始 `3eed_dataset` 复制到本地 `qa_pipeline/samples/3eed_small/`：Waymo 同一序列的 `0034_0`、`0074_0` 两帧，Drone `Outdoor_Day_fast_flight_2/000787` 和 Quad `Outdoor_Day_penno_short_loop/000787` 各一帧。每帧保留 `image.jpg`、`lidar.npy` 或 `lidar.bin`、`meta_info.json`，共 12 个原始文件；没有下载完整数据集，也没有移动/修改远程原始文件。
- 本地验证：Waymo 点云分别为 23,492×5、23,663×5；Drone 7,445×4，Quad 8,663×4；四帧 `meta_info.json` 分别含 3/3/1/1 个 `ground_info` 目标。生成 `pointcloud_topdown.svg`、`points_first_20.csv`、`annotations_summary.json` 和明确标注为项目派生数据的 `our_multi_grounding_example.json`。连同说明和生成脚本总大小约 2.9 MB；用 NumPy/JSON 读取与一张 Waymo 图像目视检查通过。
- 新增 `qa_pipeline/samples/3eed_small/README.md`，说明原始文件格式、标注字段及查看顺序。`make_previews.py` 可以从原始样本重建预览。

## 49. 核查论文正式多目标指标与本项目 5.79% 的口径差异（2026-09-21）

### 已完成动作

1. 核对论文 Table 3、multi-object 任务定义和严格评测规则。
2. 确认论文 Ours 总体报告为 Acc@0.25 32.32%、Acc@0.5 29.89%、mIoU 56.40%。
3. 检查官方默认数据 loader：公开 `ground_info` 会被展开为“一条表达 -> 一个目标框”的单目标样本。
4. 检查官方 multi loader：它依赖独立的 `waymo_multi_{train,val}_info.pkl` 及多个 `bbox3d_obj_N` 字段。
5. 确认此前 5.79% 使用的是本项目自动拼接的 908 条自制多目标数据，不是论文 Table 3 的正式数据和训练口径。
6. 新建 `OFFICIAL_MULTI_OBJECT_GAP.md`，明确三类数据、当前缺口和后续决策。
7. 完成 50 条原始 3EED 样例包：50 条记录、36 个原始场景、RGB/点云/meta、逐条记录和可交互可视化页面。
8. 新建样例包 `README.md`，说明红/蓝/灰框含义，以及“同帧多框”和“单句多目标”的区别。
9. 重新生成并检查 `index.html` 数据查看器。

### 当前状态

- 论文正式多目标效果确实明显高于 5.79%。
- 5.79% 仅作为自制多目标原型的失败基线保留，禁止再与论文 Table 3 直接对比。
- 严格复现 Table 3 的阻塞项是正式 multi-object 标注 pkl 尚未在公开原始包或当前主机中找到。

### 下一步

1. 获取作者正式 multi-object 标注，或根据论文补充材料逐项复建并验证。
2. 使用官方 `scripts_multi` 重新训练，先复现 Table 3，再导出联合目标 query/token。
3. 在 QA 中分别评测 GT 目标 token、正确预测 token和完整预测 token，分离 Grounding 与 QA 误差。
10. 用 `package_samples.py` 重新生成 `original_50.zip`，把最终 README、交互页面、50 个预览和原始场景一起纳入压缩包；完整性检查为 36 个场景、50 条记录、50 个预览。
## 50. 建立下一阶段可执行任务清单（2026-09-21）

### 已完成动作

1. 新建 `NEXT_TASKS.md`。
2. 将后续工作划分为正式多目标数据核查、官方训练复现、多目标 token 验证、空间 QA 接入和路线规划五个阶段。
3. 为每个阶段写明产出和通过标准。
4. 明确立即执行顺序：先确认正式 multi 标注，再做小样本过拟合和完整训练，之后导出 token，最后接 QA。
5. 明确论文正式多目标、公开单目标和自制 5.79% 原型结果禁止混用。

## 51. 正式多目标标注调查与官方格式复建数据集（2026-09-21）

### 已完成动作

1. 逐项核验官方多目标标注是否公开：HF `RRRong/3EED` 只有 5 个文件；用 HTTP Range 直接读 `3eed_dataset.zip` 中央目录（81,687 条目）与 `baseline_ckpt_pred.zip`（36,719 条目），确认 0 个 pkl、0 个 multi 文件；GitHub 仓库 HEAD `843a901`（2025-12-26）无 multi 数据；GitHub Issue #2 作者 2025-11-04 回复会"later"发布；全网无第三方镜像。结论：**官方 `waymo_multi_{train,val}_info.pkl` 从未发布**。
2. 顺带发现官方 `data/splits/multiwaymo_train.txt` 与 `multiwaymo_val.txt` 完全相同（MD5 均 `a075ee542c4fb2ff6928760bbe9df015`，128 行），真实多目标划分应在未发布的 pkl 内部。
3. 按仓库 loader 代码逐行整理官方多目标格式（caption、segment_name/frame_name、`bbox3d_obj_N` 7 维框、`class_obj_N`、`target_spans` 字符区间、`object_ids`；→ `positive_map [N,256]`、joint correctness 评测、Acc@25/50 + mIoU）与官方训练口径（从零、lr 1e-4、batch 12、200 epoch、lr 衰减 75/80；Table 9 的 `Object A/Object B/Relationship` 生成方式）。
4. 新建 `OFFICIAL_MULTI_ANNOTATION_SEARCH.md` 记录以上全部证据、格式与替代方案。
5. 新建 `qa_pipeline/build_official_format_multi.py`，用已发布帧的原始 caption（经人工校验的单目标标注）按官方格式重建多目标标注；沿用 QA v3 序列切分，产出 `qa_pipeline/artifacts/multi_grounding_official_fmt/waymo_multi_{train,val,test}_info.pkl` + `summary.json`：train 903（807 双目标 + 96 三目标）、val 529、test 386；三 split 序列交集为 0。
6. 本地预检全部 1,818 条记录通过：字段齐全、7 维框、span 与 `_format_caption` 结果逐字一致、类别名都在 `CLASS_MAPPINGS` 里；tokenizer 级 span 校验（`get_positive_map`）本地无 torch，留远端 smoke。
7. 为第 2/3 步准备脚本（均在 `3EED/scripts_multi/`，尚未远端运行）：`train_multi_official_fmt_smoke.sh`（官方配方 + `--debug`：annos[:128]、每轮 batch4×11=44 条）、`train_multi_official_fmt.sh`（从零、batch 12、lr 1e-4、200 epoch、lr 衰减 75/80）、`eval_multi_official_fmt.sh`（评测 + 导出 selected token）。
8. `3EED/src/grounding_evaluator.py` 多目标路径新增按类别统计（`_accumulate_class_metrics` + `print_stats` 新块）：每类 mIoU 与 Acc@0.25/0.5（Hungarian 分配后的 top-1 query）、按类别组合的联合 Acc，供第 3 步与论文 Table 3 同格式对照；只在 `last_` 前缀累计，避免多前缀重复计数。

### 当前状态

- 第 1 步（寻找正式多目标标注）完成：官方标注不存在公开版本，已有同格式替代集。
- 远程主机 SSH 全程不可达：`kex_exchange_identification: read: Connection reset by peer`（TCP 能连、域名解析正常），判断为 AutoDL 代理后端无实例，疑似实例已关机，需要用户在控制台确认。
- 第 2 步起的训练/评测都在等远程 GPU；本地新代码只做了 py_compile 与数据预检。

### 下一步

1. 远程可用后：先备份并把 `data/waymo_multi_*.pkl` 替换为 official_fmt 版本，跑 `python -m scripts_multi.smoke_multi_dataset` 做 tokenizer 级校验，再跑 smoke 训练看 loss 下降。
2. smoke 通过后用导出 NPZ + `qa_pipeline/verify_multi_export.py` 检查同一句话的不同目标是否落在不同 query。
3. 再按 `train_multi_official_fmt.sh` 完整训练，用按类别 Acc/mIoU 与 Table 3 对照（须注明数据非官方标注）。

### 更正说明

- WORK_LOG #42/#43 记录的 v3 QA val 0.4939 / test 0.5056 是**整 split**（n=5153/4219）准确率，不是子集数字；以本条为准。

## 52. 新服务器环境搭建 + 第 2 步小数据训练与 query 唯一性验证（2026-09-21）

### 环境（新 AutoDL 实例，RTX 4090 24G，<remote-training-host>）

1. 免密登录：本地 `~/.ssh/id_ed25519.pub` 写入 `/root/.ssh/authorized_keys`（密码不进任何工作区文件）。
2. 仓库同步到 `/root/3eedqa/3EED`；编译三个 CUDA 扩展（`pointnet2/_ext`、`ops/teed_pointnet/pointnet2_batch/teed_pointnet`、`ops/teed_pointnet/roiaware_pool3d`，CUDA 12.1，`-D_GLIBCXX_USE_CXX11_ABI=0`），全部 in-place 导入验证通过。
3. 依赖：`numpy==1.26.4`、`opencv-python-headless==4.10.0.84`、`open3d==0.20.0`（aliyun 镜像）；open3d 报 `GLIBCXX_3.4.30 not found`，根因是 conda 自带 libstdc++.so.6.0.29 过旧，处理方式：备份 `libstdc++.so.6.0.29` 到 `/root/autodl-tmp/3eed_data/`，用系统 `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30` 覆盖并重建软链（改动记录在此，回滚只需恢复备份）。
4. `data/roberta_base` 替换为真实权重目录（原文件是 50 字节路径指针）；`PYTHONPATH=. python data/gen_class_embeddings_local.py` 生成 `data/class_embeddings3d.npy`，shape `(485, 768)`。
5. 数据集 6,724,434,217 B 下载完整后用 `unzip` 解压到 `/root/autodl-tmp/3eed_data/3eed_dataset`（8.0G），软链 `3EED/data/3eed -> /root/autodl-tmp/3eed_data/3eed_dataset`；抽查三个 split 各 8 条 `scene_id` 对应帧，`image.jpg/lidar.npy/meta_info.json` 全部存在。

### 发现并修复的代码 bug

- `src/joint_det_dataset.py::getitem_waymo_multi` 对 `anno["pcd_path"]/["image_path"]` 原地 `replace("data/","data/3eed/")`，第二个 epoch 起会把 `data/3eed/` 再替换一次成 `data/3eed/3eed/...` 导致 `FileNotFoundError`（第 1 个 epoch 正常、第 2 个 epoch 崩）。
- 修复：替换前判断 `"data/3eed/" not in path`（幂等），本地改完已同步远端；这是上游仓库 bug，正式训练必须带此修复。

### 第 2 步结果（小数据训练测试）

1. `python -m scripts_multi.smoke_multi_dataset` 通过：train 903 / val 529 / test 386，点云 `(16384,3)`，每条目标数 2（抽查）。
2. smoke 训练（`train_multi_official_fmt_smoke.sh`，`--debug`：128 条过拟合集、每轮 11 batch×4=44 条、20 epoch）跑通全程无崩溃，~2.2s/epoch。
3. loss 明显下降：`loss_ce` 517→~185、`loss_constrastive_align` 278→120、`query_points_generation_loss` 0.10→0.006（epoch 1→20）。
4. 导出评测（`eval_multi_official_fmt.sh ckpt_epoch_last.pth val`）产出 133 个 NPZ / 529 条 / 1119 个目标；`qa_pipeline/verify_multi_export.py` exit=0：每行 `target_count ∈ {2,3}` 且等于 `object_ids_json` 长度，行内 `query_index` 全不重复，`object_token` 形状 `(count,288)` 且有限。
5. 具体样例：`batch000000.npz` 第 0 行 2 个目标 → query `[32, 47]`（不同），预测中心不同，token 范式与 `pred_size` 均已导出。
6. 导出字段：`object_token(288-D)`、`query_index`、`pred_center/pred_size`、`match_score`（置信度）、`gt_box`（仅评测用）、`utterance`、`object_ids_json`、`feature_name='last_query_features'`、`selection_policy='soft_token_unique_assignment_text_spans'`（token 由文本 span 选择，不依赖 GT 框）。
7. 注意：20 epoch debug 模型 Acc@0.25/0.5 仍为 0，属预期（数据量与轮数远小于正式配方 200 epoch / 903 条）。

### 下一步

1. 完整训练已在跑（`train_multi_official_fmt.sh`，903 条、batch 12、200 epoch、val_freq 5，log `/root/autodl-tmp/3eed_data/multi_grounding/full_train.log`），~23s/epoch，预计 2 小时内完成。
2. 完成后用 `eval_multi_official_fmt.sh` 对 val/test 评测，输出按类别 Acc@0.25/0.5、mIoU，与论文 Table 3 同格式对照（必须注明训练数据为复建集，非官方标注）。

## 53. 官方格式全量训练：磁盘写满中断与续跑 + 第 5 步准备（2026-09-21）

### 事故与处理

1. 全量训练（903 条、batch 12、200 epoch、val/save 每 5 轮）跑到 **第 130 轮存 checkpoint 时崩**：`RuntimeError: PytorchStreamWriter failed writing file data/1695: file write failed`。根因：同一时间后台在下载 Qwen2.5-7B-Instruct（15GB）到同一块 50G 数据盘，加上 26 个 checkpoint（705MB/个，共 18G）+ 数据集解压 8G，磁盘写满（df 显示 412K 剩余）。**不是代码或数据问题**。
2. 崩溃前 **第 125 轮** checkpoint 完好（705MB，torch.load 验证 model/optimizer/scheduler/epoch 齐全）；第 130 轮文件截断（82MB）已删。
3. 清理：删除除 75/100/125 外的全部中间 checkpoint（回收 ~15.5G）、删除已解压验证过的 `3eed_dataset.zip`（6.3G）、删除 smoke 运行的 5 个 checkpoint（3.5G）。清理后数据盘 25G 可用。
4. 新建 `3EED/scripts_multi/train_multi_official_fmt_resume.sh`：用 `--checkpoint_path ckpt_epoch_125.pth` **完整续跑**（代码里 `load_checkpoint` 会恢复 model+optimizer+scheduler，并把 start_epoch 设为 126），其余配方与官方一致，日志到 `official_fmt_resume`。18:46 启动，验证 lr 为 1e-6（与 75/80 衰减后的原计划一致），从 126 轮正常继续到 200。

### 第 125 轮（崩溃前）评测结果（val，n=529）

- 总：Acc@0.25 **11.53%**，Acc@0.5 **5.67%**，mIoU **12.02%**。
- 按类别（Hungarian top-1，bbs/bbf 两条口径基本一致）：车 n=870，mIoU 15.0%，Acc@0.25 25.4%，Acc@0.5 17.5%；行人 n=224，mIoU 1.7%，Acc@0.25 4.0%；truck n=17，Acc@0.25 23.5%；bus/cyclist 各 4 个，无统计意义。
- 联合全对（一句话所有目标都对）：车+车 16.7%、车+卡车 7.1%、车+行人 1.1%、行人+行人 0%。
- 轨迹（Acc@0.25）：epoch 5 → 0.4%，25 → 1.1%，45 → 5.5%，65 → ~8%，85 → ~9%，105 → ~10%，125 → 11.5%；lr 在 75/80 衰减到 1e-6 后仍在缓慢爬升，续跑 126-200 预期小幅提升。

### 第 5 步准备（已完成）

1. `main_utils.py --eval_split` 增加 train 选项（只能用于导出训练集 token，不用于报告）；`eval_multi_official_fmt.sh` 同步允许 train。
2. `build_correct_multi_qa.py` 新增 `--splits`（默认 val,test 保持旧行为；训练投影层时用 train,val,test）。
3. 上传 QA v3 数据（qa.jsonl 18,274 题、private_gt.jsonl）、neutral_bank 到远端 `/root/3eedqa/qa_pipeline/artifacts/`；上传 `train_scenario_qa.py`、`train_qa.py`、`qa_eval.py`、`build_correct_multi_qa.py`、`run_correct_multi_qa.py`。
4. Qwen2.5-7B-Instruct 已下载完成到 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`（15G）。
5. 新建一键脚本 `qa_pipeline/run_step5_official_fmt.sh`：等训练完 → 导出 train/val/test token（用 resume 的 ckpt_epoch_last.pth）→ `build_correct_multi_qa --splits train,val,test`（IoU≥0.25 筛"全目标定位正确"）→ `train_scenario_qa.py --mode lora`（Qwen bf16 + LoRA r16 + 288→1024 projector，LLM 只输入问题与隐式 token）→ val/test 评测。
6. 预估样本规模（对象集合匹配，尚未按 IoU 筛）：train 2737 / val 1583 / test 1213 道多目标题；实际训练集 = 其中 grounding 全对的部分，预计几百道量级。

### 下一步

1. 等 resume 跑完（预计 20:00 前后）；如再崩沿用同一续跑方式。
2. 直接 `nohup bash /root/3eedqa/qa_pipeline/run_step5_official_fmt.sh &`，跑完记录 Acc@0.25/0.5、mIoU（按类别，与 Table 3 同格式，注明数据为复建集）和 QA val/test 准确率。
## 54. 根据师兄反馈重新审计单目标/多目标代码并重启修正版训练（2026-09-21）

### 核查结论

1. 师兄所说“公开代码只完成可用的单目标 Grounding”基本正确：默认数据与 checkpoint 是单目标，正式 multi pkl 未发布，原 evaluator 有 `num_obj == 1` 断言。
2. 仓库并非完全没有多目标代码：已有 `waymo-multi` loader、训练脚本和多 GT Hungarian loss，但缺少可运行的完整闭环。
3. 多目标监督不能只把后续词都并入一行 positive map；必须是一目标一行 `positive_map[i]`，并与 `gt_bboxes[i]` 一一对应，再由 Hungarian matcher 分配不同 query。

### 本次发现和动作

1. 发现复建数据含五个 LiDAR 视角，而 multi loader 未像单目标 Waymo 路径那样统一到前视坐标；旧训练在方向语言上存在坐标语义不一致。
2. 19:09 温和停止仍在运行的旧版续训，保留 epoch 150 及日志，并在远端写入 `SUPERSEDED_COORDINATE_FRAME.txt`。
3. 复建标注新增 `coordinate_frame=source_lidar`，重新生成 train 903 / val 529 / test 386 三个 pkl。
4. multi loader 对带此标记的数据同时旋转点云和全部目标框；对坐标未知的未来官方 pkl 保持原行为。
5. 新增侧视真实数据断言，train/val/test 均显示 `side_view_rotation_ok`。
6. 修复总 mIoU 混入 proposal/中间层的问题，只统计最终 decoder 层。
7. 修复 `car+car` 被集合折叠成 `car` 的报告问题。
8. 20 epoch 修正版 smoke 完成：loss 下降，最终层 soft-token Top-1 逐目标 Acc@0.25 在 debug 子集约 3.2%；debug 联合指标低属预期。
9. 为避免再次写满 50G 数据盘，正式训练 checkpoint 间隔从 5 调整为 25 epoch，不改变训练优化配置。
10. 19:15 从零启动修正版 200 epoch 全量训练，远端进程 19759/19826，目录 `/root/autodl-tmp/3eed_data/multi_grounding/coordinate_fix_run`。
11. 新建 `GROUNDING_CODE_AUDIT.md`，记录官方代码边界、正确多目标监督结构、修复和旧结果处理办法。

### 结果使用规则

- 公开单目标 checkpoint 结果继续有效。
- 5.79% 与坐标修复前的 11.53% 只保留为历史诊断结果。
- 修正版完整训练与评测完成前，不把旧多目标 token 接入 QA。
## 55. 二次端到端审计：确认并修复“只监督第一个目标”（2026-09-21）

### 关键发现

1. 官方 multi loader 在遍历第 i 个目标框时仍执行 `point_instance_label[fg_mask] = 0`，所有目标点都被压成实例 0。
2. query-point loss 会根据该实例编号把 seed 分配给对应 GT，因此旧代码只对第一个目标提供正确的点级/query seed 监督；第二、第三目标虽仍进入 Hungarian 框损失和语言损失，但 query-point 辅助监督错误。
3. 这与师兄所说“当前只监督句子里的第一个目标”相符，是确定的上游多目标实现缺陷。

### 已完成动作

1. 立即停止19:15开始的错误版本全量训练，保留日志并写入 `SUPERSEDED_INSTANCE_LABEL.txt`。
2. 将 multi loader 改为 `point_instance_label[fg_mask] = i`，保留每个目标自己的实例编号。
3. 强化真实数据 smoke：检查 1,818 条标注的 positive-map 行互不重叠、box/object ID 数量一致、框中心不重复；三个 split 均检测到后续目标实例编号并通过侧视旋转测试。
4. 新建 `smoke_multi_loss.py`：合成双目标验证两个 span → 两个框 → 两个不同 query；实例编号 0/1 均进入 query-point loss并成功反向传播。
5. `smoke_multi_export.py` 再次通过，导出 query `[0,1]` 和两个 `(288,)` token。
6. 修复 `SetCriterion` 非 DDP 情况仍调用 `dist.get_world_size()` 的问题；DDP 训练行为不变。
7. 点实例修复后的20轮 smoke 完成：loss 正常下降，首次 epoch20 eval 的联合 Acc@0.25 为 2.27%、mIoU 3.61%；debug 重复评测受随机点采样影响，不作为正式指标。
8. 新建 `audit_multi_point_labels.py`，抽查各 split 前128条：有框内采样点的目标占 train 47.5%、val 22.1%、test 16.9%。无点目标仍有框/语言损失，但没有 query-point 辅助监督，后续需作为数据可见性因素分析。
9. 正式训练输出目录改为 `instance_fix_run`，必须从零开始，禁止续用坐标修复前或实例修复前 checkpoint。

## 56. 精确目标 span、确定性评测与最终重启（2026-09-21）

1. 发现复建记录曾把每个对象的整段 source caption 标为 positive；其中常包含参照车或其他物体，语言监督被污染。修改 builder：完整描述仍作为上下文，但每个目标的 positive span 只取 `neutral_query`，定位不到时退化到该对象段内的目标名词。
2. 重新生成并同步 pkl：train 903 / val 529 / test 386；span 来源计数分别为 train 1576/326、val 1021/98、test 650/151（`neutral_query/matched_noun`）。全部目标 span 非空、互不重叠。
3. 多目标评测和 288 维 token 导出统一使用一对一 Hungarian query 分配，并按每个目标 span 的 token 数归一化得分。
4. val/test 点采样改为按 `split:scan_id` 固定 SHA1 种子；重复读取测试通过。训练仍使用随机采样。
5. `run_step5_official_fmt.sh` 更新为读取本轮 `instance_fix_run` checkpoint，并写入独立的 token、QA 和 LoRA 输出目录，避免误用旧实验。
6. 20:41 用精确 span 数据从零启动 200 epoch：PID 30030/30095，日志 `/root/autodl-tmp/3eed_data/multi_grounding/instance_fix_precise_span_console.log`，运行目录 `.../instance_fix_run/.../0921_2041`。
7. 训练趋势：epoch 5 为 Acc@0.25 0、mIoU 1.62%；epoch 10 为 Acc@0.25 0.19%、mIoU 3.26%；epoch 15 为 Acc@0.25 1.13%、Acc@0.5 0.19%、mIoU 3.14%。loss 从 epoch 1 的约 122 降至后续约 30，当前没有 NaN、OOM 或进程退出。
8. 扩展点覆盖审计，同时检查原始完整点云和 16K 采样点。前 128 条中 train/val/test 目标有框内点的比例仍为 47.5%/22.1%/16.9%，且没有任何“原始有点但采样后丢失”的目标。说明限制来自原始点云可见性，而非随机抽样。
9. 代码核查确认 `--use_color` 与 `--augment_det` 在当前 multi 数据路径实际上未生效；记录为实验限制，不在训练中途改变输入配方。
10. 发现扩展 evaluator 对 bbs 使用逐目标分母、对 bbf 使用联合全对分母，同名 Acc 口径不一致。已统一为两个 head 都报告逐目标 Acc，并分别增加 `JointAll-bbs/bbf`；只影响评测，当前训练无需重启。此前周期性日志的 `Eval/acc` 是偏低的联合指标，最终结果以独立重评为准。
11. 新增 `smoke_multi_evaluator.py`，用“两目标中一目标正确”的合成例验证：bbs/bbf 的逐目标分子分母均为 1/2，联合全对均为 0/1，测试通过。
12. epoch 25 checkpoint 已成功写入（738,266,794 bytes）；旧口径周期验证为联合 Acc@0.25 4.16%、Acc@0.5 0.95%、逐目标 mIoU 6.20%，相较 epoch 5/10/15/20 总体上升。训练继续运行。
13. 收尾脚本额外检查当前训练进程已完全退出后才启动独立评测，避免 `ckpt_epoch_last` 刚写完时与训练器自带的最终验证同时占用 GPU。
14. token 导出增加同一次前向同时保存 soft-token 与 contrastive 两套一对一 query 选择结果；两套都来自同一最终层 288 维 decoder queries，不使用 GT 框选 query。现有 `object_token` 键保持 soft-token 兼容，新增 `contrastive_object_token` 等键供 val 指标对比后选择。
15. 21:02 训练到 epoch 33；旧联合口径 epoch 30 为 Acc@0.25 5.48%、Acc@0.5 1.13%，mIoU 7.52%，继续稳定上升。GPU 进程正常，数据盘剩余 16 GB。

## 57. 准确率偏低的口径拆分与原因定位（2026-09-21）

1. 训练到 epoch 97；epoch 95 旧日志为联合 Acc@0.25 8.70%、Acc@0.5 3.78%、mIoU 10.58%。
2. 新增 `recalc_saved_multi_metrics.py`，直接读取 epoch 95 保存的 529 个样本、1,119 个目标 IoU 重算：逐目标 Acc@0.25 **17.78%**、Acc@0.5 **12.60%**；联合全对 Acc@0.25 **8.70%**、Acc@0.5 **3.78%**；mIoU **10.58%**。因此 8.70% 低的一部分来自旧日志使用更严格的联合口径，但逐目标 17.78% 本身也仍偏低。
3. 当前主要限制：复建数据并非论文未发布的官方 multi 标注；只有 903 个训练组合且从随机初始化训练；val 前 128 条中仅 22.1% 目标框在原始点云内有点；当前主干实际只输入 xyz，`--use_color` 没有引入 RGB/反射率；多辆同类车的独立单目标描述合并后仍可能缺乏足够区分信息。
4. 学习率在 epoch 75/80 连续衰减后已到约 1e-6，epoch 80–95 指标接近平台期。当前 run 保留并继续到 200，以取得完整可复现结果；后续最有价值的对照是用公开单目标 checkpoint 初始化多目标模型，而不是只延长当前低学习率训练。

### 当前状态和后续动作

1. 保持本轮训练继续运行；观察 epoch 20/25 以后是否延续非零增长。
2. 训练结束后用更新后的确定性采样和归一化一对一选择器，独立评测 val/test 并导出 token。
3. 在 Grounding 评测、每目标/联合成功率和 token 对齐确认前，不启动 QA/LoRA。
4. 已准备并启动 `posttrain_instance_fix_eval.sh`：只等待本轮最终 checkpoint，校验可读后自动做确定性 val/test 评测与 token 导出；状态写到远端 `instance_fix_posttrain/status.log`，不会启动 QA/LoRA。

## 58. 改用 `others` 真实关联构造双目标数据并重新训练（2026-09-22）

### 数据修正

1. 根据用户指出的标注语义，重新读取每条 `ground_info[].others`，把其中的其他目标框作为双目标关系来源；停止使用此前由独立单目标记录任意组合的复建数据。
2. 原始 `others` 只有 `bbox_3d_other` 与 `class_other`，没有独立文本。为保证每个目标都有真实语言监督，仅保留能够按类别和 7D 框精确匹配到同帧另一条 `ground_info` 的关联；未匹配的 context-only other 不用于多目标训练。
3. 每条新样本固定包含两个目标：源 `ground_info` 目标与其匹配到的 `others` 目标；两者分别使用各自真实 caption、独立 positive span、独立 GT box 和 instance ID。
4. 去除双向重复关联后生成 train 1,137 条、val 1,300 条；训练集覆盖 842 个场景/82 个序列，验证集覆盖 903 个场景/85 个序列。
5. 原始 `others` 边共 28,823 条，其中可匹配边 4,894 条、双向去重后唯一 pair 2,437 条；未匹配的 23,929 条由于缺少目标自己的公开文本而排除。
6. 前 128 条 train/val 审计中，两个目标在原始点云及 16K 采样点中的覆盖率均为 100%，没有目标因采样丢失；这修复了旧复建集 train/val/test 仅 47.5%/22.1%/16.9% 的关键数据问题。

### 训练状态与阶段结果

1. 旧 `instance_fix_run` 在 epoch 130 停止，并写入 `SUPERSEDED_BY_OTHERS_DATA.txt`；其 checkpoint/token 不进入 QA。
2. 新数据 smoke、坐标旋转、双实例编号、不同 positive span、确定性验证采样和 token 导出均通过。
3. 2026-09-21 22:10 从零启动官方配置 200 epoch，运行目录为 `/root/autodl-tmp/3eed_data/multi_grounding/others_linked_run/Train_waymo-others-multi_Val_waymo-others-multi/0921_2210`。
4. epoch 150 验证：逐目标 contrastive Acc@0.25 **59.35%**、Acc@0.5 **36.92%**；双目标全部正确的 contrastive Joint Acc@0.25 **40.46%**、Acc@0.5 **18.54%**；mIoU **34.64%**。
5. 相对旧数据 epoch 95：逐目标 Acc@0.25 从 17.78% 升至 59.35%，联合 Acc@0.25 从 8.70% 升至 40.46%，mIoU 从 10.58% 升至 34.64%。说明 `others` 关联数据和点云可见性修正确实有效。
6. 当前主要问题从“目标没有点/数据构造错误”转为“框的精确拟合不足”：粗定位已经可用，但 Joint Acc@0.5 仍只有 18.54%。epoch 100 后指标接近平台，剩余 50 epoch 预计不会出现大幅提升。
7. contrastive query 选择在联合 Acc@0.25 上略优于 soft-token（40.46% 对 39.00%），最终 token 导出优先候选为 contrastive 选择出的 288 维 decoder token；仍需等 epoch 200 后做最终确定性评测再定。
8. 该数据集是利用公开 `others` 关系与两条真实单目标 caption 重建的双目标集，不等同于论文未公开的原生组合式多目标问句，因此暂不与论文表格直接声称同口径优劣。
9. 截止记录时训练已进入 epoch 151，进程正常，无 NaN/OOM；数据盘剩余约 7.0 GB，足够完成预定 checkpoint，但需要继续监控。

## 59. `others` 双目标 Grounding 与隐式 token QA 最终结果（2026-09-22）

### Grounding 完成情况

1. 200 epoch 训练完整结束；`ckpt_epoch_200.pth` 与 `ckpt_epoch_last.pth` 均成功保存且可加载，大小各约 705 MB。
2. 最终确定性 val 导出覆盖全部 1,300 个双目标样本。contrastive query 选择的逐目标 Acc@0.25 为 **59.38%**、Acc@0.5 为 **37.00%**；两个目标全部正确的 Joint Acc@0.25 为 **40.38%**、Joint Acc@0.5 为 **18.54%**；mIoU 为 **34.65%**。
3. epoch 150 到 200 基本无变化（59.35%→59.38%、40.46%→40.38%、34.64%→34.65%），确认低学习率阶段已进入平台。
4. train 导出为逐目标 Acc@0.25 **98.64%**、Joint Acc@0.25 **97.27%**、mIoU **63.63%**，与 val 差距很大，说明模型对 1,137 条训练 pair 明显过拟合；最终能力必须引用 val，不能引用 train。
5. 训练与验证 token 已全部导出：train 1,137、val 1,300；每个目标同时保存 soft-token 与 contrastive 两种 query 选择产生的 288 维 decoder token。

### 隐式 token QA/LoRA 结果

1. 后处理按照既有授权自动完成：用 GT IoU 只做“Grounding 是否正确”的诊断筛选（阈值 0.25），QA 输入本身仅包含问题文本与 288 维隐式 token，`geom mode none`，没有 box/center/coordinate 输入。
2. 2,437 个双目标 Grounding pair 中有 1,618 个 pair 的两个预测都通过 IoU 0.25；匹配得到 QA train 2,447 条、val 620 条，场景为 relative_location、which_is_left、closer_to_ego。
3. Qwen2.5-7B-Instruct + projector + LoRA 训练 3 epoch 完成。val 总准确率 **62.26%**，同类别同 split 的 donor token 全部打乱后为 **34.84%**；验证集按场景多数类基线约 **35.00%**。因此正常 token 相对无有效 token 提高约 27.3 个百分点，证明隐式 grounded token 整体上携带并被后端使用了空间信息。
4. 分场景：relative_location **50.0%**（打乱 20.4%，多数类 18.2%）；closer_to_ego **100%**（打乱 40.9%，多数类 54.5%）；which_is_left **56.1%**（打乱 58.5%，多数类 52.4%）。前两项有明确 token 贡献；`which_is_left` 暂时没有证明 LoRA 正确利用 token，需继续做 A/B swap 或专项训练检查。
5. 线性探针补充证据：val 全部 pair 的 token left/right 为 73.9%，两个 Grounding 都正确时为 92.8%；closer-to-ego 分别为 91.5%/97.3%。说明左/右信息存在于 token，中间瓶颈更可能位于当前 projector/LoRA 的任务利用方式。
6. 当前 LoRA 构建脚本实际读取 `object_token`，即 soft-token query 选择输出；尚未使用略强的 `contrastive_object_token`。需要增加同一 QA 子集上的 contrastive-token 对照后，才能决定最终接入哪个 token。
7. 当前 62.26% 是“两个目标均正确 Grounding 后”的诊断上限实验，并非完整端到端 QA；真实端到端还要把约 40.38% 的 val 双目标 Grounding 成功率计入。
8. 目前问答输出为固定空间答案模板，已经验证 token→projector→LLM 链路，但还不能据此声称完成自由自然语言问答或路线规划。

## 60. QA v4 数据与关系建模改造（2026-09-22）

### 已完成的代码修改

1. `build_correct_multi_qa.py` 新增 `--token-variant soft|contrastive`，token、预测中心和预测尺寸使用同一 selector 的输出，避免用 soft box 筛选却接入 contrastive token 等不一致情况；v4 默认使用 contrastive。
2. 新增训练集 A/B 成对反转补全：对 `relative_location`、`which_is_left`、`closer_to_ego` 交换对象、token role、问题和答案；如果数据中已经存在反向记录则不重复添加。
3. 对 `which_is_left` 与 `closer_to_ego` 的训练答案增加严格 A/B 数量平衡断言。
4. 额外生成独立 `swap_eval` 验证集，只用于检查交换 A/B 后预测是否随物理对象正确翻转，不混入普通验证指标或训练集。
5. `train_scenario_qa.py` 新增 `RelationalTokenProjector`：为 A/B/C 加可学习 role embedding，用一层 8-head Transformer 在 288 维 token 间先做关系交互，再经共享 288→1024→LLM hidden projector；没有加入坐标、box 或距离输入。
6. 训练入口扩展为 `projector`、`lora`、`eval` 三种模式；v4 先冻结 LLM 单独训练关系模块与 projector，再由该 checkpoint 初始化 LoRA 阶段。
7. 新建 `run_others_v4_relation.sh`，依次执行 contrastive QA 构建与断言、关系 Projector 训练、LoRA 训练、同类别 donor-token 打乱评测、A/B swap 评测并生成最终摘要。
8. 两个修改后的 Python 文件已通过本地语法编译；本机 Python 未安装 torch，因此涉及 3D IoU 与模型的运行检查转到远程 agiclass 环境执行。

## 61. QA v4 关系模型最终实验结果（2026-09-22）

### 数据与运行检查

1. contrastive selector 下共有 2,437 个双目标 pair，其中 1,631 个 pair 的两个目标均达到 IoU 0.25；得到原始 matched QA train 2,441 / val 615。
2. 训练集补充 673 条 `which_is_left` A/B 反转和 496 条 `closer_to_ego` A/B 反转，最终 train 3,610、val 615；两个选择任务在训练集中均严格 A/B 各半。
3. 生成独立 `swap_eval` 615 条，普通 val 不加入反转副本，保持与旧实验相近的评测规模。
4. 远程语法、导入、数据数量、token identity、答案平衡全部通过；最终 QA 公共输入逐条检查不含 box、center、coordinate、distance 或 answer_key。
5. 最佳 LoRA checkpoint 确认包含 role embedding、关系 Transformer 与 288→1024→3584 projector 权重；最佳为 LoRA epoch 1，val loss 0.065。GPU 已释放。

### 分阶段训练结果

1. 关系 Projector 单独训练三轮，val loss 0.1009→0.0764→0.0670；冻结 LLM 时 val 准确率已达 **78.86%**：relative_location 63.58%、which_is_left 92.50%、closer_to_ego 100%。
2. 用最佳 Projector 初始化 LoRA，最佳 checkpoint 的 val 总准确率 **80.00%**：relative_location **65.74%**、which_is_left **92.50%**、closer_to_ego **100%**。
3. 相比旧 soft-token 独立 Projector+LoRA 的 62.26%，v4 提高 **17.74 个百分点**；旧 `which_is_left` 为 56.10%，v4 为 92.50%。
4. 同类别、同 split donor token 全部替换后准确率降至 **36.91%**：relative_location 22.22%、which_is_left 50.63%、closer_to_ego 56.49%；接近该 val 的约 35% 多数类基线，说明 80% 主要来自正确 token，而不是文本模板或标签偏置。
5. A/B swap eval 总准确率 **80.33%**：relative_location 65.74%、which_is_left 93.75%、closer_to_ego 100%；与普通 val 基本一致。
6. 对普通与 swap 的同一样本逐一检查逻辑反转：总体预测一致率 **91.87%**（565/615）；relative_location 88.27%、which_is_left 92.50%、closer_to_ego 100%。普通与反转两题同时正确为 77.24%。
7. swap eval 的 exact sentence 为 45.85%，低于解析准确率 80.33%，原因是生成句式与反转 GT 模板的表面措辞不同；关系解析键正确，因此主要报告语义准确率与逻辑一致率。

### 当前判断

1. QA 数据的 A/B 对称监督和关系 Projector 是有效修改，已经修复左右任务不能稳定使用 token 的问题。
2. LoRA 相对 Projector-only 仅从 78.86% 提升到 80.00%，当前主要能力来自关系模块；继续增加 LoRA 轮数价值较低，且 epoch 2 val loss 已升至 0.077，出现过拟合。
3. 结果仍是 IoU 0.25 筛选后的 token 能力诊断；完整端到端 QA 需要保留全部 val pair、不使用 GT IoU 筛选再评测。
4. 当前远程数据盘剩余约 4.7 GB；后续新增大 checkpoint 前应先清理已废弃 run，避免写满磁盘。

## 62. 全预测 token 的端到端 QA 评测（2026-09-22）

### 实现与检查

1. `build_correct_multi_qa.py` 新增 `--eligibility all`，允许取消 GT IoU 筛选，将所有可对齐的 Grounding 预测 token 送入 QA；新增 `--keep-without-donor`，避免端到端主评测因缺少打乱对照 donor 而丢样本。
2. GT IoU、每个目标 IoU 和 Grounding 是否通过阈值仅写入 private diagnostic 文件，不进入问题、object_refs 或模型输入。
3. 新增 `analyze_end_to_end.py`，可复现统计总准确率、分场景结果、Grounding 正确/错误子集、最小目标 IoU 分桶、标签分布与多数类基线。
4. 使用已训练好的 v4 contrastive relation-projector + LoRA checkpoint，不重新训练；对所有能与 QA v3 对齐的 val pair 直接生成答案。

### 最终结果

1. 现有 1,300 个 val Grounding pair 中有 508 个唯一 pair 能与 QA v3 的对象集合对齐，共生成 **1,638 条**端到端 QA：relative_location 924、which_is_left 463、closer_to_ego 251。其余 pair 当前缺少对应 QA v3 问题，未混入分母。
2. 不做 GT IoU 筛选的端到端总准确率为 **57.08%**（935/1,638），parse rate 100%；按场景为 relative_location **37.23%**、which_is_left **73.65%**、closer_to_ego **99.60%**。
3. 当前标签分布的加权多数类基线为 **35.29%**，端到端结果高 21.79 个百分点。
4. 508 个唯一 pair 中，180 个双目标均达到 IoU 0.25，328 个未达到。对应 QA 行中：Grounding 正确子集 **80.00%**（492/615），Grounding 未通过子集 **43.30%**（443/1,023）。
5. Grounding 未通过时，relative_location 降到 **21.83%**，which_is_left 为 **63.70%**，closer_to_ego 为 **99.17%**。说明精确的二物体相对关系最依赖正确 Grounding；距离自车任务即使预测框 IoU 低，token 中仍保留很强的径向位置信息。
6. 按两个目标中较小 IoU 分桶：<0.10 为 43.28%，0.10–0.25 仅 4 条不作结论，0.25–0.50 为 87.38%，≥0.50 为 71.72%。后两桶场景组成不同，因此不能把该非单调结果直接解释为 IoU 越高 QA 越差。
7. 可复现结果保存在远端 `runs/others_v4_end_to_end/eval_val.json` 与 `analysis.json`；GPU 已释放。

### 结论与剩余数据问题

1. 当前完整链路在可对齐 QA 子集上的真实结果为 **57.08%**；此前 80.00% 应明确标为“Grounding 正确条件下的 token QA 上限实验”。
2. 下一项数据工作应直接从 1,300 个 val multi-grounding pair 的 GT box 与两条 caption 生成同结构 QA，使所有 pair 都进入端到端分母，而不是继续依赖只覆盖 508 个 pair 的旧 QA v3 交集。

## 63. 场景级可变目标、高难 QA 与严格评测新要求（2026-09-22）

### 用户确认的新目标

1. Grounding 数据从每条固定 2 个目标改为场景级可变目标数，数量由当前场景的真实标注决定。
2. QA 升级为多目标、多步空间推理，每条答案必须是一个完整句子。
3. 除空间关系和距离外，加入车辆行驶路径规划问题。
4. 最终主指标使用严格合取：关系关键词、严格 `(A, relation, B)` 三元组、完整标准句、A/B 交换反转、Grounding 条件能力和无 GT 筛选端到端能力必须分别统计；一条样本只有所有适用项同时通过才算正确。

### 数据与架构审计

1. 全量 Waymo 公开 metadata 共 5,409 帧，`ground_info` 目标数分布为：1 目标 3,656 帧、2 目标 1,482 帧、3 目标 231 帧、4 目标 32 帧、5 目标 8 帧；最大值为 5。
2. 现有 Hungarian loss、`box_label_mask`、多目标 evaluator 和 token 导出已按 `target_count` 工作；当前主要硬限制是组合文本/positive map 固定 256 token，QA role embedding 固定最多 3 个角色。
3. 多目标组合 caption 最长约 237 个英文词，256 个 RoBERTa subword 可能截断后面目标。新场景模型将文本/soft-token head/positive map 统一扩展到 512，并对每个目标的 positive span 做非空断言。
4. 3EED metadata 有当前点云、物体框、caption 和 pose，但没有唯一导航终点、HD lane graph、交通规则或完整动态轨迹条件。因此仅依赖 3EED 可生成“固定局部前向目标 + 当前障碍物”的决定性局部避障路径；不将它冒充为完整道路路线规划。完整规划需补 Waymo Motion/Map、ego 历史状态和导航 goal。
5. “场景目标数”第一版严格定义为当帧所有有独立 caption 的 `ground_info` 目标。`others` 中未匹配的 context box 没有自己的 referring expression，不伪造文本监督；后续可作为无语言的 detector context token 单独加入。

### 已实现与实测

1. 新增 `build_scene_multi_grounding.py`，在服务器全量生成 `waymo_scene_multi_{train,val}_info.pkl`；train 2,701 帧/3,687 目标，val 2,708 帧/3,794 目标，全部 5,409 帧和 7,481 个有 caption 目标均进入数据，数量范围 1–5。
2. 3EED 新增 `waymo-scene-multi` loader，文本 tokenizer、positive map 和 soft-token head 统一可配置为 512；Grounding evaluator 取消 contrastive 评测中写死的 256 padding。
3. 真实数据 loader smoke 通过：train/val 全量 positive span 非空且不重叠，目标数 1–5，多实例点标签、侧向 LiDAR 坐标旋转和 val 确定性通过。
4. 新增 `smoke_scene_model.py`，对真实 4 目标样本完成一次 GPU forward + Hungarian/contrastive loss + backward；`positive_map=(1,132,512)`、`sem_scores=(1,256,512)`、`query_features=(1,256,288)`，loss 83.6046，4090 峰值显存约 1,936 MB。
5. 新增 `build_scene_reasoning_qa.py`，生成 5,518 条一句话 QA：train 2,656 / val 2,862；含 3,506 条多干扰物 A/B 关系题、271 条两步关系链、271 条距离排序后关系题和 1,470 条局部路径规划题。
6. 新增 `strict_scene_qa_eval.py`，只有关系关键词集、全部严格三元组集、完整标准句、一句话结构、规划动作、A/B swap 全通过才计 `language_all_ok`；端到端还要求所有被引用目标 Grounding 通过。
7. 用 5,518 条 canonical answer 作 oracle prediction 时，关键词、三元组、完整句、一句话、规划动作、A/B swap 和严格合取指标均为 100%，证明数据生成与评测器自洽；Grounding 条件/端到端指标需等新模型 token 导出后才有真实数值。
8. 场景级 Grounding 已在远程 RTX 4090 启动 100 epoch 训练，使用旧 checkpoint 部分初始化；旧权重中 1,020 个张量直接载入、14 个文本 head 张量由 256 扩展到 512，没有跳过不兼容权重。真实训练已通过 epoch 10 并继续运行，无 OOM/NaN。
9. epoch 10 首次验证：`PerTargetContrast@0.25=71.56%`、`@0.5=41.70%`；要求场景内全部目标同时正确时，soft-token head 为 `48.46%/20.33%`，contrastive head 为 `50.88%/21.32%`（对应 IoU 0.25/0.5）。这是中途结果，不能作为最终模型结论。
10. 总体 Joint 指标会被 1 目标场景占比影响，因此 evaluator 新增按场景目标数分组的严格指标：分别报告 N=1、2、3、4、5 时“所有目标均超过 IoU 阈值”的准确率。该修改不重启当前训练，将用于后续验证与最终 checkpoint 评测。

### 本阶段验收口径

1. `language_all_ok = keyword_ok ∧ triples_ok ∧ canonical_sentence_ok ∧ one_sentence_ok ∧ planning_ok ∧ swap_ok`，其中只合取该样本适用的检查项。
2. `strict_end_to_end_ok = language_all_ok ∧ all_referenced_targets_grounded`；任何一个被问题引用的目标定位失败，整条样本即失败。
3. Grounding 条件准确率只在所有被引用目标均定位正确的子集上计算，用于定位 QA 后端能力；端到端准确率在完整数据上计算，作为真实系统主结果。
4. 3EED 当前只支持基于当帧障碍物和固定前向终点的局部避障动作。完整道路路线规划还需要 Waymo Motion/Map、ego 历史状态和导航终点，不能从现有 3EED 标注可靠生成。

## 64. 场景级 Grounding 训练中途审计（2026-09-22）

### 当前实际训练内容

1. 当前运行只训练 3EED Grounding，不训练 QA、LoRA 或路径规划。输入是一个场景的点云和该场景 1–5 个目标各自的真实 referring expression；模型产生 256 个候选 query，通过 Hungarian matching 同时监督所有目标的中心、尺寸、GIoU、soft-token 分类和 query/text contrastive alignment。
2. 每个被匹配目标最终对应一个不同的 288 维 decoder query feature；后续训练结束后才导出这些 token，并接入 Projector + LLM QA。

### 运行状态与中期结果

1. 审计时训练运行至 epoch 28/100，RTX 4090 占用约 12.7/24.6 GB、GPU 利用率约 86%、温度 57°C；日志中无 OOM、NaN、Traceback 或进程退出。
2. epoch 10→20 的验证指标均提高：PerTarget Contrastive Acc@0.25 从 71.56% 到 73.38%，Acc@0.5 从 41.70% 到 43.41%；JointAll Contrastive Acc@0.25 从 50.88% 到 55.38%，Acc@0.5 从 21.32% 到 25.71%。因此训练没有出现立即发散。
3. 数据结构检查通过：train/val 场景无交叉，全部 target span 合法，没有重复 object ID 或重复目标框。
4. 数据盘剩余约 4.6 GB；当前 run 的日志、TensorBoard 和 prediction 文件约 16 MB，预计最终两个约 738 MB checkpoint 可以保存，但需避免同时产生大量历史 checkpoint。

### 已发现的问题

1. 目标数严重不平衡。train 中 N=1/2/3/4 分别为 1,858/709/125/9，N=5 为 0；val 中 N=1/2/3/4/5 为 1,798/773/106/23/8。当前架构能够处理五目标，但训练集没有任何五目标监督，不能声称已学会五目标 Grounding。
2. epoch 20 的 contrastive joint 结果也反映该问题：`car+car` Acc@0.25 为 73.20%，`car+car+car` 为 36.73%，`car+car+car+car` 为 5.88%，`car+car+car+car+car` 为 0%。四/五目标样本数很少，单次百分比波动也很大。
3. 当前总体 JointAll 会受到大量单目标场景主导，必须在最终离线评测中同时报告 N=1…5 分组指标。分组 evaluator 已写入磁盘，但正在运行的 Python 进程在修改前已加载旧类，因此要在 checkpoint 上重新执行一次评测才会显示新指标。
4. 当前 `save_freq=100`，运行期间在 epoch 100 前没有恢复 checkpoint；若主机中断会丢失中途训练，而且不能选择 epoch 20/30 等可能更好的模型。当前进程不重启以免丢失已有进度，后续训练入口需改成保留一个滚动 latest checkpoint 和一个 best checkpoint。
5. 当前评测使用标注中的 target text span 选择相应 query，衡量的是“给定 referring expression/span 的多目标 Grounding”。面向任意自然语言 QA 的完整系统还需要问题解析或无需 GT span 的 query 选择，不能把当前指标直接当作自由问答端到端结果。

### 处理决定

1. 当前 100 epoch 运行继续作为不均衡数据上的场景级基线，因为 epoch 10→20 仍在稳定提升；不在没有 checkpoint 的 epoch 28 强制中断。
2. 最终 checkpoint 必须补做 N=1…5 分组评测并导出 token。随后建立目标数平衡的训练版本：按 N 分层采样或过采样 N=3/4/5，并重新划分少量五目标场景到 train，同时保证 scene/sequence 不泄漏；该版本与当前自然分布基线分开报告。

## 65. 多目标文字标注来源与 positive span 复核（2026-09-22）

### 原始标注是否真的包含多个目标描述

1. 重新读取全部原始 `meta_info.json`：5,409 个场景的 `ground_info` 合计包含 7,481 条独立目标记录；每条记录各自带有 `class`、`caption`、`bbox_3d`、`bbox_2d_proj` 和 `others`。因此 N=2…5 不是从一条描述复制出来的，而是同一帧本来就有多条独立 `ground_info`。
2. 构建记录与原始文件逐项比较：5,409/5,409 场景存在，目标数量、7,481 条 caption、class、3D box 均为 0 个不一致；场景内没有重复 caption。
3. 机器核对证明“我们没有改写或串错原始 caption/box”；原始 caption 本身的颜色、朝向等语义仍只能通过人工看 RGB 抽查，不能用字段相等自动证明。

### 找到并修复的真实问题

1. v1 的 caption/box 来源虽然正确，但 positive noun span 抽取不完全正确：至少 68 条选择了 `There is/There are ...` 后面的环境物体词，另有 17 条因词表未覆盖而插入了类别词。例如目标描述开头是 `silver truck`、数据类别为 `car` 时，旧词表可能选择后文环境中的 `sedan`。
2. 扩充 released target noun 词表，覆盖 `range rover`、`truck`、`tool cart`、`MPV`、`Jeep`、`loader`、`excavator`、`camper van` 等原始用词；`license plate` 保留为部件式 referring expression 的目标 span。
3. 重新生成 v3 并用两套审计器复查：7,481/7,481 均使用原文里的 target noun span，人工插入类别 0 条，`There is/There are` 后的明确环境词误选 0 条，caption/class/box 不一致仍为 0。v1→v3 共改变 165 个 span，其中至少 68 个是明确环境词误选、17 个是人工类别 span，其余主要是将 `vehicle` 等宽泛词换为原文更早、更具体的 `Range Rover/truck/...`。
4. v3 已替换仓库和服务器磁盘上的后续数据文件。正在运行的旧进程在启动时已经把 v1 annotations 加载到内存，因此当前自然分布 baseline 仍包含上述旧 span；运行已到 epoch 92，不中断，完成后明确标为 v1 baseline，后续平衡训练使用 v3。

### 可视化抽查

1. 保存 N=1、2、3、4、5 各一个原始场景，并将原始 `bbox_2d_proj` 直接画在 RGB 上，以 A–E 标出目标；完整原始 caption、2D box、3D box 见 `qa_pipeline/artifacts/scene_multi_audit/EXAMPLES.md`。
2. 人工查看所选五张图：N=1 行人、N=2 两名行人、N=3 两名行人加一辆车、N=4 四辆 SUV、N=5 五辆并排车辆的框、类别、颜色和相对排列均与对应 caption 基本一致。
3. 扩大抽查时发现一条明确的疑似原始标注噪声：`waymo/8133434654699693993_1162_020_1182_020/0060_0` 的 Object B 投影框圈中黑色 pickup，但同一条原始 `ground_info` caption 的主语是 `white sedan`，并把 black pickup 写成环境中的 behind 对象。转换器没有串行，它忠实复制了这组原始 caption/box；问题来自发布标注本身或其生成过程。
4. 因此“7,481 条 caption/class/box 零字段不一致”只能证明来源复制正确，不能宣称 7,481 条语义全部正确。下一轮平衡训练前必须先对全部多目标场景做 RGB 语义质量筛查，把疑似 caption-box 不一致样本加入 quarantine；当前已完成的 baseline 需标注为含原始标注噪声。

## 66. 场景级 Grounding 第一轮训练完成与 v3 复评（2026-09-22）

1. 自然分布 baseline 已完成 100 epoch 并正常退出，无 OOM、NaN 或 Traceback；成功保存 `ckpt_epoch_100.pth`（744,484,438 bytes）和 `ckpt_epoch_last.pth`（744,487,298 bytes）。
2. 训练进程启动时已把 v1 annotations 载入内存，所以该 checkpoint 的训练监督仍含旧 positive-span 问题。训练完成后使用磁盘上的修正 v3 val annotations 和新版 evaluator 独立复评同一 `ckpt_epoch_100.pth`。
3. v3 复评总体结果：PerTarget Contrastive Acc@0.25 **75.43%**、Acc@0.5 **46.49%**、mIoU **44.94%**；要求场景内全部目标同时正确的 JointAll Contrastive Acc@0.25 **55.49%**、Acc@0.5 **26.15%**。
4. contrastive Joint 分目标数结果（Acc@0.25/0.5）：N=1 **78.20%/45.72%**（1,798）；N=2 **59.51%/27.81%**（773）；N=3 **42.45%/21.70%**（106）；N=4 **0%/0%**（23）；N=5 **0%/0%**（8）。
5. 结论：模型已证明同一次 forward 可监督和输出不同数量目标，对一至三目标有可测能力；四、五目标尚未学会。总体 55.49% 被大量 N=1 场景明显抬高，不能用该总体值声称多目标任务已经解决。
6. 可复现摘要保存为 `artifacts/results/scene_multi_v1_train_v3_eval.json`。下一轮应在完成 caption-box 语义筛查后，使用 v3、按目标数量分层采样，并确保 train 中包含足够 N=4/5 场景。

## 67. 远程 PKL 可读 JSON 预览（2026-09-22）

1. 将当前生效的 v3 `waymo_scene_multi_{train,val}_info.pkl` 转成四个可直接用编辑器查看的 JSON 文件，保存在远程 `/root/3eedqa/3EED/data/scene_multi_previews/`。
2. `waymo_scene_multi_train_preview_50.json` 和 `waymo_scene_multi_val_preview_50.json` 分别包含各 split 前 50 条完整记录。
3. `examples_N1_to_N5.json` 保存 N=1、2、3、4、5 各一条完整记录；`examples_N1_to_N5_compact.json` 是便于人工阅读的精简版，仅保留 split、scene、组合 caption、各目标原始描述、类别、positive span/word 和 3D box。
4. 四个文件均重新用 JSON parser 读取验证通过，记录数分别为 50、50、5、5。

## 68. QA 直接位置文本泄漏修复（2026-09-23）

1. 用户人工查看后指出旧 QA 问题直接出现目标相对位置。本次逐条检查确认：`scene_reasoning_qa_v1` 的 5,518/5,518 条 `question` 都拼入了原始 3EED caption，并至少命中一个 `left/right/front/behind/lower/upper/located/positioned/situated/middle` 类空间提示；公开 `object_refs` 也全部带 `description`。因此 v1 不能作为隐式 token 空间推理证据。
2. 进一步发现两个角色顺序泄漏：`rank_then_relation` 把第二近目标固定重排为 A，`local_path_planning` 把最近障碍物固定重排为 A。即使删除 caption，大模型也可能靠角色位置猜答案。
3. 修改 `build_scene_reasoning_qa.py`：公开问题只保留 A–E 角色和通用任务模板，公开 `object_refs` 只保留 role 与用于 token 对齐的 opaque object ID；caption、category 和 GT geometry 全部移到 private GT。角色顺序由 scene ID 稳定打乱，并按真实角色生成排序和规划答案，不再把正确对象提前放到 A。
4. 生成 `artifacts/local_qa/scene_reasoning_qa_v2_noleak/`，仍为 5,518 条：train 2,656、val 2,862；题型数量与 v1 相同。自动审计为：公开 caption 命中 0、公开禁用字段 0、坐标格式 0、left/right/front/behind 提示词 0。
5. 排序题答案角色分布为 A/B/C/D/E = 76/93/87/13/2；规划最近障碍物角色分布为 687/703/75/4/1，不再是固定 A。少数 D/E 来自四、五目标场景本身很少。
6. 用 canonical answer 作为 oracle prediction 回灌 `strict_scene_qa_eval.py`：关系关键词、严格三元组、完整句、一句话、规划动作、A/B 交换和严格语言合取均为 100%。Grounding 条件与端到端结果仍为空，因为尚未把新场景 checkpoint 的真实预测 token 对齐到 v2。
7. v2 修复了 LLM prompt 的直接文本泄漏，但最终 288 维 decoder token 仍是语言条件化特征。正式结论前需做当前 caption token、中性 caption token、同 query 视觉特征、同类别打乱 token 和无 token 消融，避免把编码进 token 的 caption 位置词误判为点云几何推理。
8. `build_scene_token_qa.py` 的同类别打乱对照已改为从 private `grounding_inputs` 读取 category；公开 `object_refs` 不再为打乱实验暴露类别，训练代码仍只读取 token、role 和通用 question。

## 69. 无泄漏场景 QA 正式训练启动（2026-09-23）

1. 用户确认开始训练。远程 RTX 4090 空闲，Qwen2.5-7B-Instruct、场景 Grounding `ckpt_epoch_100.pth` 和 v2 无泄漏 QA 均存在；启动前数据盘剩余约 3.2 GB。
2. 检查发现场景级 1–5 目标的 288 维 token 尚未导出，不能直接训练 Projector/LoRA。新增可恢复的一键任务 `run_scene_qa_v2_noleak.sh`：依次导出 train/val token、校验、构建 token QA、训练关系 Projector、训练 LoRA、评测同类别打乱 token。每阶段成功后写 `.done`，中断后可从未完成阶段继续。
3. `verify_multi_export.py` 已扩展为支持 1–5 目标、单目标 batch 的无 target 轴格式、soft/contrastive 两套 token、行内 query 唯一性和 288 维有限值检查；在旧双目标 train 导出 1,137 场景/2,274 目标上回归通过。
4. 修复 `train_scenario_qa.py` 的评测入口：检测到 private GT 含 `canonical_answer` 时，改用 `strict_scene_qa_eval.py`，输出关系关键词、严格三元组、完整标准句、一句话、规划动作、A/B 交换、Grounding 条件和端到端严格指标；旧 QA 仍使用原解析器。
5. 远程代码完成 `py_compile`、严格评测器 import 和 shell 语法检查。后台任务 PID 2966，于 2026-09-23 12:55 启动；状态目录 `/root/autodl-tmp/3eed_data/multi_grounding/scene_qa_v2_status/`。首阶段为 train split token 导出，共 2,701 场景、901 batch；启动后 GPU 进程正常，占用约 1.56 GB，无即时异常。
6. 本次训练只使用 `scene_reasoning_qa_v2_noleak` 的通用问题文字和 A–E 隐式 token。原始 caption、类别、GT box/center 和关系标签不进入 LLM prompt；GT 几何仅保留在 private 文件用于标签与评测。

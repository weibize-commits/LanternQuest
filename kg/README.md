# LanternQuest KG 原型

本目录实现灯彩研究的知识图谱开发骨架。它以宣纸 NSCEF-ICHKG 的“本体—规则—多模态—分层展示”思路为参照，增加证据与任务状态分离、路径证据义务、三值状态、意图锁和可追溯同意。

当前生成物是 `candidate` 开发图，不是领域专家确认的正式知识图谱。来源的 `rights_status` 仍为 `unreviewed`，因此不得对外发布原媒体或把候选规则写成传统标准。

## 目录

- `docs/LANTERN_KG_ARCHITECTURE.md`：架构、相对创新和实验假设。
- `docs/NSCEF_TO_IPER_EXTENSION.md`：在宣纸 NSCEF-ICHKG 基础上的具体扩展、公式和证伪实验。
- `ontology/lanternquest.ttl`：OWL/Turtle 概念与关系骨架。
- `schemas/lantern-kg.schema.json`：图交换格式 JSON Schema。
- `schemas/neural-alignment.schema.json`：冻结神经编码器输出的跨模态对齐候选格式。
- `schemas/asr-transcript-pilot.schema.json`：本地 ASR 先导的模型、音频、时间戳与审核边界格式。
- `tools/build_seed_graph.py`：从任务候选和来源清单生成开发图。
- `tools/build_multimodal_graph.py`：把全部来源构造成内容寻址的多模态资料图，并合并首任务语义证据。
- `tools/neurosymbolic_gate.py`：用来源、范围、冲突、权利和安全规则门控神经候选。
- `tools/validate_graph.py`：检查引用完整性、证据链和决策证据义务。
- `data/dev_seed_graph.json`：生成的首任务开发图。
- `data/dev_multimodal_graph.json`：全量多模态开发图；媒体资产不等同于已审核的工艺实体。
- `data/chinese_clip_first_task_pilot.json`：Chinese-CLIP 本地 CUDA 图文编码先导。
- `data/dev_multimodal_neurosymbolic_graph.json`：合并神经候选与符号门控结果的开发图。
- `data/faster_whisper_field07_pilot.json`：Field07 访谈的本地 ASR 时间戳候选。
- `data/faster_whisper_field07_diagnostic.json`：相对现有粗转录的诊断结果，不作为金标准准确率。
- `data/dev_multimodal_neurosymbolic_asr_graph.json`：加入22个待审核音频转录片段后的开发图。
- `../annotations/eval/multimodal_gold_pilot_v0.csv`：12个图文候选与22个音频候选的双人标注队列。
- `../annotations/eval/MULTIMODAL_GOLD_PROTOCOL_V0.md`：标签、独立双标、裁决和报告边界。
- `data/nscef_extension_mechanisms_v0_results.json`：路径证据义务、未知状态和意图锁机制检查。
- `models/model_registry_v0.json`：候选模型、执行位置和实际运行状态。
- `rules/neurosymbolic_policy_v0.json`：开发冻结的神经分数融合与符号门控策略。
- `tests/test_validate_graph.py`：独立校验测试。

## 使用

```powershell
.\.venv\Scripts\python.exe kg\tools\build_seed_graph.py
.\.venv\Scripts\python.exe -m kg.tools.build_multimodal_graph
.\.venv-multimodal\Scripts\python.exe -m kg.tools.run_chinese_clip_pilot
.\.venv\Scripts\python.exe -m kg.tools.integrate_alignment_pilot
.\.venv-multimodal\Scripts\python.exe -m kg.tools.run_faster_whisper_pilot
.\.venv\Scripts\python.exe -m kg.tools.evaluate_asr_pilot
.\.venv\Scripts\python.exe -m kg.tools.integrate_asr_pilot
.\.venv\Scripts\python.exe -m kg.tools.build_multimodal_annotation_queue
.\.venv\Scripts\python.exe -m kg.tools.analyze_multimodal_annotations
.\.venv\Scripts\python.exe -m kg.tools.run_nscef_extension_mechanisms
.\.venv\Scripts\python.exe kg\tools\validate_graph.py kg\data\dev_seed_graph.json
.\.venv\Scripts\python.exe kg\tools\validate_graph.py kg\data\dev_multimodal_graph.json
.\.venv\Scripts\python.exe -m pytest -q kg\tests
```

每次冻结正式图谱前，都应先完成来源权利审核、领域专家复核、冲突裁决和版本签名。开发图只能用于数据管线、界面与算法联调。

基础 `dev_multimodal_graph.json` 有 1,782 个节点和 1,806 条关系，覆盖 873 个来源记录和 861 个唯一内容资产。图文与音频先导合并后的 `dev_multimodal_neurosymbolic_asr_graph.json` 有 1,822 个节点和 1,978 条关系，包括12个图文对齐候选、22个音频转录候选和相应模型/规则节点；所有神经候选仍待人工审核，0条自动提升为遗产事实。只有首任务的7条主张属于已经回看来源的语义知识，因此不能把该图报告为已经完成神经抽取的正式多模态 KG。

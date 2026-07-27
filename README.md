# emotionX

**关系条件化的情感反应引擎** —— 同一句用户输入，在不同关系设定下产生方向相反的情感反应。

```
用户输入： 我想要你
关系设定： 情侣    →  亲和↑ 唤起↑ 效价↑
关系设定： 陌生人  →  戒备↑ 唤起↑ 效价↓ 主导↑（设界）
```

不是两条规则，是同一条规则在不同参照系下的两个解。设计说明见
[`docs/architecture.md`](docs/architecture.md)。

---

## 状态

正在从 v1（情绪分类 + 状态机）重构为 v2（关系条件化评价 + 多通道状态）。
当前 `main` 上是 v1 的完整可用实现，v2 正在按里程碑推进。

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 仓库初始化 | ✅ |
| M1 | v2 核心类型（RelationalFrame / UserMove / 6 通道状态） | ⬜ |
| M2 | 关系性 appraisal 引擎（失配、互补性、交叉抑制、习惯化） | ⬜ |
| M3 | 反事实测试套件 | ⬜ |
| M4 | 安全域架构 | ⬜ |
| M5 | 表达层 + 显示层 + 动作门控 | ⬜ |
| M6 | WebUI 集成平台 | ⬜ |
| M7 | 外部记忆系统适配器 | ⬜ |
| M8 | L1 训练适配到回归输出 | ⬜ |

---

## 快速开始

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,annotate]"
.venv/bin/python -m pytest tests/ -q
```

训练与评估需要额外依赖（生产运行时**不装** torch，见下）：

```bash
uv pip install --python .venv/bin/python -e ".[training]"
```

### 多轮对话 demo（无需模型，L1 走规则桩）

```bash
.venv/bin/python -m affect.cli --persona warm_companion
```

实时打印内部状态。元命令：`:state` `:prompt` `:event task_failed=1` `:persona <name>` `:reset` `:quit`

### 状态轨迹评审（调参的主要依据）

```bash
.venv/bin/python eval/trajectory_review.py
```

10 个典型剧本 × 每个 persona，图输出到 `artifacts/trajectories/`。

### 本地标注站

```bash
.venv/bin/python annotate/server.py --import data/raw/seed_pool.jsonl --source seed
.venv/bin/python annotate/server.py          # → http://127.0.0.1:8077
```

只监听 `127.0.0.1`。键盘流：`1`–`4` 选标签 · `↑↓←→` 调数值 · `Enter` 提交 · `S` 跳过 · `U` 撤销。

标注完成后按标签**均衡**挑选评估集并冻结（macro-F1 与先验分布无关，
均衡抽样在同样标注预算下统计功效高一倍）：

```bash
curl -X POST 'http://127.0.0.1:8077/api/golden/select?per_class=60'
```

### 训练 L1

```bash
# 阶段一：通用情感感知预训练（EWECT 自动下载）
.venv/bin/python training/stage1_pretrain.py --datasets ewect --epochs 3

# 阶段二：策略标签微调
.venv/bin/python training/stage2_finetune.py --stage1 artifacts/l1_stage1 \
    --annotations data/exports/stage2_train.jsonl

# 导出 ONNX + int8 量化 + 一致性校验 + 延迟基准
.venv/bin/python training/export_onnx.py --model-dir artifacts/l1_stage2 --out artifacts/l1_onnx
```

阶段一实测：`nghuyong/ernie-3.0-nano-zh`（17.9M）在 SMP2020-EWECT 上
3 epoch / 87s（M5 MPS）→ dev macro-F1 **0.7275**。

### 接入

```python
from affect import AffectPipeline, ConversationEvent

pipeline = AffectPipeline(model_dir="artifacts/l1_onnx", store_backend="redis")
result = pipeline.process_turn(
    session_id="u-123",
    user_utterance="还是不行，一样的报错",
    last_agent_reply="请再试一次",
    event=ConversationEvent(user_repeated_query=True, latency_ms=6200, turn_count=4),
    persona_name="steady_medical",
)
affect_prompt, generation_params = result.as_tuple()
```

---

## 目录

| 路径 | 说明 |
|---|---|
| `src/affect/state_machine.py` | **状态层 —— 系统核心**，纯计算无神经网络 |
| `src/affect/expression.py` | 状态 → 行为指令 + 生成参数 |
| `src/affect/safety.py` | 安全边界（硬编码，不可配置） |
| `src/affect/perception.py` | 感知层推理：ONNX 与规则桩两个实现 |
| `src/affect/tokenization.py` | 纯 Python WordPiece（生产环境不装 transformers） |
| `config/` | 评价规则、表达模板、persona 参数、标注指南 |
| `annotate/` | 本地标注站（FastAPI + SQLite） |
| `training/` | 两阶段训练 + 蒸馏 + ONNX 导出 |
| `eval/` | 轨迹评审、感知层指标 |
| `docs/architecture.md` | 设计说明 |

**运行时依赖刻意保持极简**：生产环境不安装 torch。
感知层走 ONNX Runtime，其余是纯 Python。

## 设计约束

三条在代码里强制、不可通过配置绕过的约束：

1. **状态层的失败模式是情绪传染**——用户情绪恶化时关切必须上升，agent 自身效价只轻微下降。
   写成了启动时的可执行断言（`StateMachine.assert_no_contagion()`）。
2. **评估集只能是人工标注的真实会话**。加载器与评估脚本都会拒绝含开源/蒸馏数据的评估集——
   用蒸馏数据评估，测的是"学生像不像老师"，教师的系统性偏差会被完美继承且完全隐形。
3. **危机识别独立于模型**，由规则+关键词层承担，优先级高于全部逻辑。

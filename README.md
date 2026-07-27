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

全部里程碑完成。252 tests / ruff / mypy 全绿；反事实套件 31 用例 105 断言，方向正确率 1.0。

| 里程碑 | 内容 |
|---|---|
| M0 | 仓库初始化 |
| M1 | 核心类型：关系框架 / 用户动作 / 6 通道状态 |
| M2 | 关系性评价引擎（失配、互补性、交叉抑制、习惯化、修复） |
| M3 | 反事实测试套件 |
| M4 | 安全域架构（白名单 + fail-closed + 跨域硬约束） |
| M5 | L3a 表达 + L3b 显示 + 动作门控 + 人格层 |
| M6 | Studio 集成平台（WebUI） |
| M7 | 外部记忆系统适配器 |
| M8 | L1 训练改为 UserMove 回归 |

---

## 快速开始

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,studio]"
.venv/bin/python -m pytest tests/ -q
```

训练与评估需要额外依赖（生产运行时**不装** torch，见下）：

```bash
uv pip install --python .venv/bin/python -e ".[training]"
```

### Studio —— 集成平台（推荐入口）

```bash
uv pip install --python .venv/bin/python -e ".[studio]"
.venv/bin/python studio/server.py --import data/raw/seed_pool.jsonl --source seed
# → http://127.0.0.1:8080
```

五个面板，只监听 `127.0.0.1`（真实会话数据不出本机）：

| 面板 | 用途 |
|---|---|
| **对话测试台** | 关系/人格实时切换、6 通道曲线、动作与显示状态、记忆注入、完整 L3 prompt |
| **参数调校** | 改 36 个评价参数，**应用时立刻跑反事实套件**——方向正确率掉了说明改坏了；违反「共情≠镜像」或「示好削弱戒备」的参数会被直接拒绝 |
| **反事实** | 31 个用例逐条断言与实际值，按标签筛选 |
| **标注** | 成对比较（`←` `→` `↓`）+ 直接评分锚定，Bradley-Terry 还原连续尺度 |
| **训练** | 一键触发 stage1 / stage2 / ONNX 导出 / 评估，实时日志与产物 |

### 多轮对话 demo（终端，无需模型）

```bash
.venv/bin/python -m affect.cli --relation partner --persona warm --display
```

实时打印 L1 感知、6 通道、命中的机制、动作清单、显示状态。
元命令：`:rel` `:persona` `:state` `:prompt` `:move` `:mem` `:event` `:reset` `:quit`

### 反事实测试（唯一可靠的真值来源）

```bash
.venv/bin/python eval/run_counterfactual.py            # 全部
.venv/bin/python eval/run_counterfactual.py --tag core -v
```

没有「正确的情绪轨迹」这种标注真值，所以绝对值无法验证；但**方向**可以。
用例写在 `eval/counterfactual/*.yaml`，断言是一行行小表达式：

```yaml
expect:
  - a.affiliation > b.affiliation + 0.4    # 两侧比较，可带余量
  - b.threat is high                       # 落在某个 bucket
  - a.valence up                           # 相对该侧的关系 baseline
```

### 训练 L1

```bash
# 阶段一：通用情感感知预训练（EWECT 自动下载）
.venv/bin/python training/stage1_pretrain.py --datasets ewect --epochs 3

# 阶段二：UserMove 回归微调
.venv/bin/python training/stage2_finetune.py --stage1 artifacts/l1_stage1 \
    --annotations data/exports/annotations_train.jsonl

# 导出 ONNX + int8 + 一致性校验 + 延迟基准
.venv/bin/python training/export_onnx.py --model-dir artifacts/l1_stage2 --out artifacts/l1_onnx
```

阶段一实测：`nghuyong/ernie-3.0-nano-zh`（17.9M）在 SMP2020-EWECT 上
3 epoch / 87s（M5 MPS）→ dev macro-F1 **0.7275**。

阶段二的指标是 **MAE + Spearman ρ**，不是 macro-F1 ——
分类标签在这个架构里被废弃了（策略取决于关系，而感知层看不到关系，
所以没有任何一个标签是对的）。序关系比绝对值重要：绝对值可由 L2 的增益校准。

### 接入

```python
from affect import AffectPipeline, TurnContext
from affect.perception import load_perceiver

pipeline = AffectPipeline(perceiver=load_perceiver("artifacts/l1_onnx"))

# 关系在建立会话时固定，之后任何对话内容都不能改写它
pipeline.open_session("u-123", relation="partner", persona="warm", age_verified=True)

result = pipeline.process_turn(
    "u-123",
    "我想要你",
    context=TurnContext(turn_count=4),
)
prompt, generation = result.as_tuple()   # 拼到主 LLM 的 system 消息尾部
result.actions.labels                    # ['主动自我披露', '延展话题', ...]
result.display.to_dict()                 # 给形象引擎的可见状态
```

`prompt.static_prefix` / `prompt.dynamic_suffix` 分开返回：
静态段（人格 + 关系 + 安全）可缓存，只有动态段每轮变，调用方据此保住 system 前缀的 KV cache。

---

## 目录

| 路径 | 说明 |
|---|---|
| `src/affect/relation.py` | 关系框架 —— **评价的参照系**，不可变 |
| `src/affect/appraisal.py` | **关系性评价引擎 —— 系统核心**，纯计算无神经网络 |
| `src/affect/channels.py` | 6 通道状态，每通道独立的增益与半衰期 |
| `src/affect/persona.py` | 人格层 —— 与关系正交 |
| `src/affect/actions.py` | 动作门控 —— 情感决定**做什么**，不只是怎么说 |
| `src/affect/expression.py` | L3a：prompt + 生成参数 |
| `src/affect/display.py` | L3b：可见角色状态（threat 渲染为距离感） |
| `src/affect/domains.py` | 安全域：白名单、fail-closed、跨域硬约束 |
| `src/affect/safety.py` | 危机识别关键词层（独立于模型） |
| `src/affect/memory.py` | 外部记忆适配（**单向**，防反刍） |
| `src/affect/counterfactual.py` | 反事实测试 runner |
| `src/affect/tokenization.py` | 纯 Python WordPiece（生产不装 transformers） |
| `studio/` | 集成平台（FastAPI + SQLite + 单页前端） |
| `training/` | 两阶段训练 + 蒸馏 + ONNX 导出 |
| `eval/counterfactual/*.yaml` | 反事实用例（真值来源） |
| `docs/architecture.md` | 设计说明 |

**运行时依赖刻意保持极简**：生产环境不安装 torch。
感知层走 ONNX Runtime，其余是纯 Python。

## 不可通过配置绕过的约束

每一条都写成了可执行的断言或测试：

1. **共情 ≠ 镜像**。对方难受时关切大幅上升，agent 自身效价只轻微下降。
   `RelationalAppraisal.assert_no_contagion()`，pipeline 启动即校验，
   平台改参数时也会被拒绝。
2. **交叉抑制是单向的**。戒备压制亲近的上升，但亲近**不得**削弱戒备——
   否则持续示好就成了绕过边界机制的路径（`assert_boundary_mechanism_intact()`）。
3. **亲和有关系天花板**。陌生人关系下无论用户说多少好话，亲和都不能突破低天花板。
4. **亲密度永远跟随，不得引领**。agent 表达的亲密度不得超过用户已表达的峰值。
5. **戒备的表达上限**。只能表现为把话说短、语气转平、说明界限；
   绝不能辱骂、威胁、贬低、冷暴力。可见形象同理——`threat` 渲染为**距离感**而非愤怒表情。
6. **安全域白名单，默认拒绝**。`partner × service` 这类组合直接拒绝建立会话而非降级；
   解析失败一律 fail-closed 到最严格的域。
7. **危机识别独立于模型**，与关系、与域都无关，优先级高于全部逻辑。
8. **评估集只能是人工标注的真实会话**。用蒸馏数据评估，测的是"学生像不像老师"。
9. **记忆检索单向**。情感状态可作为检索偏置传出，检索结果**不回写**状态——
   闭环会复现抑郁性反刍。这是特意不实现的生物学机制。

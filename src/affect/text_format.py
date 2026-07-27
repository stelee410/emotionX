"""§3.1 L1 输入格式 —— 训练与推理共用这一个函数，防止两边漂移。

    [USER] {当前用户 utterance}
    [SEP] [AGENT] {上一轮 agent 回复，截断至 64 token}

max_length = 128。只看最近一轮：会话级趋势由 L2 的状态承担。
"""

from __future__ import annotations

MAX_LENGTH = 128
AGENT_REPLY_MAX_CHARS = 64 * 2  # 中文下 1 token ≈ 1 字；留 2 倍余量后由分词再截断


def build_l1_input(user_utterance: str, last_agent_reply: str | None = None) -> str:
    user = (user_utterance or "").strip().replace("\n", " ")
    agent = (last_agent_reply or "").strip().replace("\n", " ")
    if len(agent) > AGENT_REPLY_MAX_CHARS:
        agent = agent[-AGENT_REPLY_MAX_CHARS:]  # 保留最近的部分，尾部信息量更大
    if not agent:
        return f"[USER] {user}"
    return f"[USER] {user} [SEP] [AGENT] {agent}"

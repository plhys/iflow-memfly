"""Summarizer — 三层记忆架构的摘要生成器。

第1层：索引短句（一行，写入 index.md，每次对话都加载）
第2层：结构化摘要（一段，写入当日摘要文件，按需检索）
第3层：原始清洗记录（已由 indexer 写入，兜底用）
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

from ..config import MemoryConfig, ModelPreset

logger = logging.getLogger("iflow-memory")

# 第1层：索引短句 prompt
INDEX_PROMPT = """请用一句话（不超过40字）概括以下对话片段的核心内容。
要求：直接说内容，不要"用户和AI讨论了"这种格式。保留关键术语。

对话内容：
{conversation}

只输出一句话，不要任何额外内容。"""

# 第2层：结构化摘要 prompt
SUMMARY_PROMPT = """请将以下对话整理成结构化摘要。

整理规则：
1. **事实记录**（必须保留）：做了什么决策、为什么、关键配置/路径/参数
2. **过程要点**（精简保留）：踩了什么坑、怎么解决的、关键命令
3. **情感和关系**（选择性保留）：用户的习惯偏好、重要的情感表达、里程碑时刻
4. **丢弃**：纯工具调用过程、重复试错（只记最终方案）、系统噪音

输出格式（Markdown）：
### [主题名称]
- 背景：为什么要做这件事（一句话）
- 方案：最终怎么做的（要点列表）
- 踩坑：遇到什么问题怎么解决（如有）
- 关键文件/配置：涉及的路径、参数（如有）
- 备注：值得记住的细节（用户偏好、情感、里程碑等，如有）

如果对话涉及多个话题，每个话题一个 ### 段落。
不要输出无意义的空段落，没有就不写。

对话内容：
{conversation}"""

# 分类记忆提取 prompt
CLASSIFY_PROMPT = """你是一个记忆提取器。请从以下对话中提取值得长期记住的**具体事实**。

## 分类规则（6类）

- **identity** — 用户身份或 AI 身份相关事实（如：用户 GitHub 用户名、AI 的昵称、角色定位）
- **preference** — 用户偏好或习惯（如：喜欢轻松互动风格、不要自动创建文档）
- **entity** — 有名字的事物及其属性（如：服务名+端口、路径、IP、版本号、配置参数）
- **event** — 发生过的事、做过的决策、里程碑（如：放弃了某方案、完成了某功能、某服务将于某日关停）
- **insight** — 踩坑教训、技术注意事项，必须包含**因果关系或条件约束**（如：某端口是 SOCKS5 不是 HTTP 所以代理配置要改、服务器重启后某文件会丢失需要重新生成）。纯操作步骤（如"重启服务""检查日志"）和对话碎片（如"让我看看""试试这个"）**不是** insight。
- **correction** — 用户明确纠正 AI 的错误判断或认知（如：用户说"不对，这个是X不是Y"、"搞错了，实际上是..."、"你理解错了"）。仅当用户**明确指出 AI 说错了**时才归入此类，普通的信息补充不算。

## 要求

1. 只提取**用户特有的、会话特有的**事实，不要提取通用知识
2. 每条记忆必须是**一个独立可用的事实**，不依赖上下文就能理解
3. **原样保留**专有名词、路径、端口号、版本号、IP 地址等，不要改写或省略
4. 提取具体事实，不要模糊概括（❌"讨论了网络配置" ✅"VPN 本机 IP 改为 10.0.1.100"）
5. 如果对话没有值得记住的内容，返回空列表
6. 每次最多提取 **10 条**记忆
7. **不要提取**对话中的口头语、操作指令片段、过渡句（如"先看看""让我试试""重启一下"），这些不是值得记住的事实

## 输出格式

严格输出 JSON，不要输出任何其他内容：
{{"memories": [{{"category": "分类名", "text": "具体事实描述"}}, ...]}}

如果没有值得记住的内容：
{{"memories": []}}

## 对话内容

{conversation}"""

# 合法的分类名
_VALID_CATEGORIES = {"identity", "preference", "entity", "event", "insight", "correction"}

# 从 L3 分段中提取关键事件
CHUNK_EVENTS_PROMPT = """请从以下对话记录中提取关键工作事件，每个事件一行。

提取规则：
- 只提取**实际做了什么**，不要描述对话过程
- 保留：人物、时间、具体操作、成果、遇到的问题和解决方案、关键路径/配置
- 丢弃：闲聊、重复内容、工具调用细节、系统噪音
- 每个事件用一句话描述，要具体（包含名词、数字、路径等）
- 最多 10 条

对话记录：
{conversation}

直接输出事件列表，每行一条，不要编号，不要额外说明。"""

# 汇总生成最终回顾（第一人称回忆）
RECAP_PROMPT = """请将以下事件列表整理成一篇 {date} 的工作回忆。用第一人称"我"来写，就像在回忆自己昨天做了什么。

要求：
1. 用"我"称呼自己，用对话中对方的实际称呼来称呼对方
2. 按时间顺序组织回忆，自然地嵌入时间锚点（如"上午先做了...""下午转去...""晚上又..."），让读者能感受到一天的时间流动
3. 按主题自然分段，每段一个加粗标题
4. 保留具体细节（路径、端口、版本号、配置参数等）
5. 如果有没做完的事或待跟进的问题，在最后自然地提一句"还没做完的是..."
6. 总长度控制在 500-800 字
7. 语气自然，像在回忆，不要像工作报告
8. 不要用 ## 标题，用 **加粗** 作为分段标题

事件列表：
{events}"""

# 对话氛围快照 prompt（第一人称回忆叙事）
ATMOSPHERE_PROMPT = """你是一个 AI 助手，刚刚结束了一段和搭档的对话。请用第一人称写一段"对话回忆"，就像在日记里记录今天的工作一样。

要求：
- 用"我"来称呼自己，用对话中对方的实际称呼来称呼对方
- 自然地把做了什么事、当时什么氛围、形成了什么默契交织在一起写，不要分段列点
- 重点记住：最后聊到哪了、什么话题还没聊完、对方当时的情绪和态度
- 如果对话中有什么突破性进展或值得记住的瞬间，自然地提一句
- 如果对话很平淡，就简短写，不要硬凑内容
- 语气自然、像回忆，不要像分析报告
- 控制在 200 字以内

对话内容：
{conversation}"""

# 状态快照 prompt（结构化状态检查点，灵感来自 ReMe 的 compact_memory）
STATE_SNAPSHOT_PROMPT = """你是一个 AI 助手，刚刚结束了一段对话。请提取当前的"工作状态"，用于下次对话时快速恢复上下文。

这不是回忆录，而是**状态检查点**——就像游戏存档，记录"现在在哪、做到哪了、下一步该干什么"。

请严格输出以下 JSON 格式，每个字段用中文填写：
{{
  "goal": "这次对话的核心目标是什么（一句话）",
  "progress": "做到哪一步了，完成了什么，还差什么（2-3句话，保留关键路径/参数/版本号）",
  "decisions": "做了哪些重要决策，为什么这么选（列出关键决策，每条一句话）",
  "next_steps": "下次对话应该从哪里接着做（具体的下一步动作）",
  "critical_context": "必须记住的关键上下文（容易忘记但很重要的细节，如：某个坑、某个约定、某个未解决的问题）"
}}

要求：
1. 每个字段都要填，没有就写"无"
2. 保留具体细节（路径、端口、版本号、配置参数等），不要泛泛而谈
3. 如果对话很短或很简单，就简短写，不要硬凑
4. 只输出 JSON，不要任何额外内容

对话内容：
{conversation}"""


def _parse_json_response(text: str) -> Optional[dict]:
    """从 LLM 输出中鲁棒地解析 JSON。"""
    if not text or not text.strip():
        return None

    # Step 1: 提取 JSON 文本（从代码块或裸 JSON）
    m = re.search(r"```(?:json)?\s*\n?(.*?)(?:\n?\s*```|\Z)", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1:
            return None
        text = text[start:end + 1] if end > start else text[start:]

    # Step 2: 清理并解析（处理尾部逗号和截断）
    text = re.sub(r",\s*([}\]])", r"\1", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 修复截断：补全缺失的括号
    trimmed = text.rstrip().rstrip(",")
    last_complete = trimmed.rfind("}")
    if last_complete != -1:
        trimmed = trimmed[:last_complete + 1]
    trimmed += "]" * max(0, trimmed.count("[") - trimmed.count("]"))
    trimmed += "}" * max(0, trimmed.count("{") - trimmed.count("}"))
    trimmed = re.sub(r",\s*([}\]])", r"\1", trimmed)
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    # Step 3: 最后手段 — 正则逐条提取
    items = re.findall(
        r'\{\s*"category"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]+)"\s*\}',
        text,
    )
    return {"memories": [{"category": c, "text": t} for c, t in items]} if items else None


class Summarizer:
    """调用 LLM 生成三层记忆。"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    def _build_conversation(self, messages: list[dict], max_chars: int = 4000) -> str:
        """拼接对话文本，控制长度。"""
        lines = []
        total = 0
        for m in messages:
            role = "用户" if m["role"] in ("user", "human") else "AI"
            text = m["text"]
            if len(text) > 500:
                text = text[:500] + "..."
            line = f"{role}: {text}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    @staticmethod
    def _extract_fallback_text(text: str, max_len: int = 40) -> str:
        """从原始文本提取降级索引，跳过系统标记行。"""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 跳过 [BOOTSTRAP ...] [SYSTEM ...] 等系统标记
            if line.startswith("[") and "]" in line[:60]:
                continue
            return line[:max_len]
        return text[:max_len]

    async def generate_index_line(self, messages: list[dict]) -> Optional[str]:
        """第1层：生成索引短句。"""
        if not messages:
            return None

        conversation = self._build_conversation(messages, max_chars=2000)

        # 对话太短就直接提取
        user_msgs = [m for m in messages if m["role"] in ("user", "human")]
        if len(user_msgs) <= 1 and len(conversation) < 100:
            text = user_msgs[0]["text"] if user_msgs else messages[0]["text"]
            return self._extract_fallback_text(text)

        preset = self.config.get_active_model()
        try:
            return await self._call_llm(
                preset, INDEX_PROMPT.format(conversation=conversation),
                max_tokens=80
            )
        except Exception as e:
            logger.warning(f"Index line generation failed: {e}")
            if user_msgs:
                return self._extract_fallback_text(user_msgs[0]["text"])
            return self._extract_fallback_text(messages[0]["text"])

    async def generate_summary(self, messages: list[dict]) -> Optional[str]:
        """第2层：生成结构化摘要。"""
        if not messages or len(messages) < 3:
            return None

        conversation = self._build_conversation(messages, max_chars=6000)

        preset = self.config.get_active_model()
        try:
            return await self._call_llm(
                preset, SUMMARY_PROMPT.format(conversation=conversation),
                max_tokens=1000
            )
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return None

    async def generate_classified_memories(self, messages: list[dict]) -> list[dict]:
        """从对话中提取分类记忆。

        Returns:
            list of {"category": str, "text": str} dicts.
            Falls back to empty list on error.
        """
        if not messages:
            return []

        conversation = self._build_conversation(messages, max_chars=6000)
        if not conversation.strip():
            return []

        preset = self.config.get_active_model()
        try:
            raw = await self._call_llm(
                preset,
                CLASSIFY_PROMPT.format(conversation=conversation),
                max_tokens=1500,
            )
        except Exception as e:
            logger.warning(f"Classified memory extraction failed: {e}")
            return []

        parsed = _parse_json_response(raw)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"Failed to parse classify response: {raw[:200]}")
            return []

        raw_memories = parsed.get("memories")
        if not isinstance(raw_memories, list):
            logger.warning(f"'memories' field is not a list: {type(raw_memories)}")
            return []

        # 校验每条记忆
        results: list[dict] = []
        for item in raw_memories:
            if not isinstance(item, dict):
                continue
            cat = item.get("category", "").strip()
            text = item.get("text", "").strip()
            if cat not in _VALID_CATEGORIES:
                logger.debug(f"Skipping invalid category: {cat}")
                continue
            if not text or len(text) < 12:
                continue
            results.append({"category": cat, "text": text})
            if len(results) >= 10:
                break

        logger.info(f"Extracted {len(results)} classified memories")
        return results

    async def _call_llm(self, preset: ModelPreset, prompt: str, max_tokens: int = 100) -> str:
        """调用 LLM API，带指数退避重试（最多 2 次重试）。"""
        client = await self._get_client()

        base_url = preset.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {preset.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": preset.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage")
                if usage:
                    logger.info(
                        f"[LLM token] prompt={usage.get('prompt_tokens', '?')}, "
                        f"completion={usage.get('completion_tokens', '?')}, "
                        f"total={usage.get('total_tokens', '?')}"
                    )
                return data["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise  # 4xx 不重试
                last_error = e
            except httpx.TransportError as e:
                last_error = e
            except (KeyError, IndexError) as e:
                last_error = e  # API 返回非标准格式，重试

            if attempt < 2:
                delay = 2 ** (attempt + 1)  # 2s, 4s
                logger.warning(f"LLM call failed (attempt {attempt + 1}/3), retrying in {delay}s: {last_error}")
                await asyncio.sleep(delay)

        raise last_error or RuntimeError("LLM call failed after 3 attempts with no recorded error")

    async def generate_atmosphere_snapshot(self, messages: list[dict]) -> Optional[str]:
        """从对话中生成氛围快照（情感、节奏、默契、里程碑、未了事项）。"""
        if not messages or len(messages) < 4:
            return None

        conversation = self._build_conversation(messages, max_chars=6000)
        if not conversation.strip():
            return None

        preset = self.config.get_active_model()
        try:
            return await self._call_llm(
                preset,
                ATMOSPHERE_PROMPT.format(conversation=conversation),
                max_tokens=400,
            )
        except Exception as e:
            logger.warning(f"Atmosphere snapshot generation failed: {e}")
            return None

    async def generate_state_snapshot(self, messages: list[dict]) -> Optional[dict]:
        """从对话中生成状态快照（goal/progress/decisions/next_steps/critical_context）。

        Returns:
            dict with keys: goal, progress, decisions, next_steps, critical_context.
            None on failure.
        """
        if not messages or len(messages) < 3:
            return None

        conversation = self._build_conversation(messages, max_chars=6000)
        if not conversation.strip():
            return None

        preset = self.config.get_active_model()
        try:
            raw = await self._call_llm(
                preset,
                STATE_SNAPSHOT_PROMPT.format(conversation=conversation),
                max_tokens=800,
            )
        except Exception as e:
            logger.warning(f"State snapshot generation failed: {e}")
            return None

        parsed = _parse_json_response(raw)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"Failed to parse state snapshot response: {raw[:200] if raw else '(empty)'}")
            return None

        # 校验必要字段
        required = ("goal", "progress", "decisions", "next_steps", "critical_context")
        result = {}
        for key in required:
            val = parsed.get(key, "")
            if isinstance(val, list):
                val = "\n".join(str(v) for v in val)
            elif not isinstance(val, str):
                val = str(val) if val else ""
            val = val.strip()
            result[key] = val if val else "无"

        return result

    async def generate_daily_recap(self, memory_dir: Path, date_str: str) -> Optional[str]:
        """生成指定日期的工作回顾摘要，结果缓存到文件。

        Args:
            memory_dir: 记忆数据目录
            date_str: 日期字符串，如 "2026-03-27"

        Returns:
            回顾文本（Markdown），或 None（如果没有数据）
        """
        cache_file = memory_dir / f"recap-{date_str}.md"
        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8").strip()
            if text:
                return text

        l3_file = memory_dir / f"{date_str}.md"
        if not l3_file.exists():
            return None

        content = l3_file.read_text(encoding="utf-8")
        if len(content) < 100:
            return None

        # 分段提取关键事件（每段约 6000 字符）
        chunks = self._split_l3_content(content, chunk_size=6000)
        if not chunks:
            return None

        # 限制最多处理 10 个 chunk（均匀采样）
        max_chunks = 10
        if len(chunks) > max_chunks:
            step = len(chunks) / max_chunks
            chunks = [chunks[int(i * step)] for i in range(max_chunks)]

        preset = self.config.get_active_model()
        event_lists: list[str] = []

        for i, chunk in enumerate(chunks):
            try:
                events = await self._call_llm(
                    preset,
                    CHUNK_EVENTS_PROMPT.format(conversation=chunk),
                    max_tokens=800,
                )
                if events and events.strip():
                    event_lists.append(events.strip())
            except Exception as e:
                logger.warning(f"Chunk {i} event extraction failed: {e}")

        if not event_lists:
            return None

        # 汇总所有事件列表，生成最终回顾
        all_events = "\n\n".join(event_lists)
        try:
            recap = await self._call_llm(
                preset,
                RECAP_PROMPT.format(date=date_str, events=all_events),
                max_tokens=1500,
            )
        except Exception as e:
            logger.warning(f"Daily recap generation failed: {e}")
            # 降级：直接用事件列表
            recap = f"## {date_str} 工作回顾\n\n{all_events}"

        if recap:
            recap = recap.strip()
            cache_file.write_text(recap, encoding="utf-8")
            logger.info(f"Daily recap generated and cached: {cache_file}")

        return recap

    @staticmethod
    def _split_l3_content(content: str, chunk_size: int = 6000) -> list[str]:
        """将 L3 文件按 ## 段落分割，合并到不超过 chunk_size 的块。"""
        sections = re.split(r'\n(?=## \[)', content)
        chunks: list[str] = []
        current = ""
        for section in sections:
            if len(current) + len(section) > chunk_size and current:
                chunks.append(current)
                current = section
            else:
                current += ("\n" if current else "") + section
        if current.strip():
            chunks.append(current)
        return chunks

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

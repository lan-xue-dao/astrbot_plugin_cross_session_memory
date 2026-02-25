"""
AstrBot 跨会话记忆插件（基于向量嵌入）
使用 AstrBot 内置的 Embedding Provider 实现智能记忆存储和检索
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger
from astrbot.api import AstrBotConfig


@dataclass
class MemoryRecord:
    """记忆记录"""
    id: str
    content: str
    session_id: str
    group_name: str
    sender_name: str
    role: str  # "user" 或 "assistant"
    timestamp: str
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "session_id": self.session_id,
            "group_name": self.group_name,
            "sender_name": self.sender_name,
            "role": self.role,
            "timestamp": self.timestamp,
            "embedding": self.embedding
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        return cls(**data)


@dataclass
class MemoryGroup:
    """记忆组"""
    name: str
    session_ids: Set[str]
    max_history: int = 50
    memories: List[MemoryRecord] = field(default_factory=list)

    def add_memory(self, record: MemoryRecord):
        """添加记忆记录"""
        self.memories.append(record)
        # 保持历史记录在限制内
        if len(self.memories) > self.max_history:
            self.memories = self.memories[-self.max_history:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "session_ids": list(self.session_ids),
            "max_history": self.max_history,
            "memories": [r.to_dict() for r in self.memories]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGroup":
        return cls(
            name=data["name"],
            session_ids=set(data.get("session_ids", [])),
            max_history=data.get("max_history", 50),
            memories=[MemoryRecord.from_dict(r) for r in data.get("memories", [])]
        )


@register(
    "astrbot_plugin_cross_session_memory",
    "lanxuedao",
    "让 Bot 能够跨群组/会话共享记忆，使用向量嵌入实现智能检索",
    "1.0.0",
    "https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory"
)
class CrossSessionMemoryPlugin(Star):
    """跨会话记忆插件（基于向量嵌入）"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 记忆组存储
        self.session_to_group: Dict[str, MemoryGroup] = {}
        self.memory_groups: Dict[str, MemoryGroup] = {}

        # 数据存储路径
        self.data_dir = os.path.join("data", "plugin_data", "astrbot_plugin_cross_session_memory")
        self.data_file = os.path.join(self.data_dir, "memory_data.json")

        # 自动保存任务
        self._save_task: Optional[asyncio.Task] = None
        self._running = True

        # 嵌入服务
        self.embedding_provider = None
        self.use_embedding = False

        # 初始化
        self._init_config()
        self._load_data()
        self._init_embedding()

    def _init_config(self):
        """初始化配置"""
        self.enabled = self.config.get("enabled", True)
        memory_groups_config = self.config.get("memory_groups", [])
        global_settings = self.config.get("global_settings", {})

        self.auto_save_interval = global_settings.get("auto_save_interval", 60)
        self.include_sender_info = global_settings.get("include_sender_info", True)
        self.cross_group_prefix = global_settings.get("cross_group_prefix", "[{group_name}] {sender}: ")
        self.use_embedding = global_settings.get("use_embedding", False)
        self.top_k = global_settings.get("top_k", 5)
        self.embedding_threshold = global_settings.get("embedding_threshold", 0.3)

        # 确保 memory_groups_config 是列表类型
        if not isinstance(memory_groups_config, list):
            logger.warning(f"[跨会话记忆] memory_groups 配置格式错误，预期列表，实际类型: {type(memory_groups_config)}")
            memory_groups_config = []

        # 初始化记忆组
        for group_config in memory_groups_config:
            if not isinstance(group_config, dict):
                logger.warning(f"[跨会话记忆] 跳过无效的记忆组配置")
                continue

            group_name = group_config.get("group_name", "default")
            session_ids = group_config.get("session_ids", [])
            max_history = group_config.get("max_history", 50)

            if group_name in self.memory_groups:
                existing_group = self.memory_groups[group_name]
                existing_group.session_ids.update(session_ids)
                existing_group.max_history = max_history
            else:
                group = MemoryGroup(
                    name=group_name,
                    session_ids=set(session_ids),
                    max_history=max_history
                )
                self.memory_groups[group_name] = group

            for sid in session_ids:
                self.session_to_group[sid] = self.memory_groups[group_name]

        logger.info(f"[跨会话记忆] 已加载 {len(self.memory_groups)} 个记忆组")

    def _init_embedding(self):
        """初始化嵌入服务"""
        if not self.use_embedding:
            logger.info("[跨会话记忆] 未启用嵌入模式，使用简单文本匹配")
            return

        try:
            # 尝试获取嵌入服务提供商
            embedding_provider_name = self.config.get("embedding_provider", "")
            if embedding_provider_name:
                self.embedding_provider = self.context.get_provider_manager().get_provider(embedding_provider_name)
                if self.embedding_provider:
                    logger.info(f"[跨会话记忆] 成功加载嵌入服务: {embedding_provider_name}")
                else:
                    logger.warning(f"[跨会话记忆] 未找到嵌入服务: {embedding_provider_name}")
                    self.use_embedding = False
            else:
                logger.info("[跨会话记忆] 未配置嵌入服务提供商，使用简单文本匹配")
                self.use_embedding = False
        except Exception as e:
            logger.error(f"[跨会话记忆] 初始化嵌入服务失败: {e}")
            self.use_embedding = False

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的嵌入向量"""
        if not self.use_embedding or not self.embedding_provider:
            return None

        try:
            result = await self.embedding_provider.get_embeddings_async([text])
            if result and len(result) > 0:
                return result[0]
        except Exception as e:
            logger.error(f"[跨会话记忆] 获取嵌入向量失败: {e}")

        return None

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        if not emb1 or not emb2 or len(emb1) != len(emb2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _load_data(self):
        """从文件加载数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                memory_groups_data = data.get("memory_groups", {})
                if not isinstance(memory_groups_data, dict):
                    logger.warning(f"[跨会话记忆] memory_groups 数据格式错误")
                    return

                for group_name, group_data in memory_groups_data.items():
                    if group_name in self.memory_groups:
                        loaded_group = MemoryGroup.from_dict(group_data)
                        self.memory_groups[group_name].memories = loaded_group.memories

                logger.info(f"[跨会话记忆] 已从文件加载记忆数据")
        except Exception as e:
            logger.error(f"[跨会话记忆] 加载数据失败: {e}")

    def _save_data(self):
        """保存数据到文件"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            data = {
                "memory_groups": {
                    name: group.to_dict()
                    for name, group in self.memory_groups.items()
                }
            }

            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug("[跨会话记忆] 记忆数据已保存")
        except Exception as e:
            logger.error(f"[跨会话记忆] 保存数据失败: {e}")

    async def _auto_save_loop(self):
        """自动保存循环"""
        while self._running:
            await asyncio.sleep(self.auto_save_interval)
            self._save_data()

    async def initialize(self):
        """插件初始化"""
        if self.enabled and self.auto_save_interval > 0:
            self._save_task = asyncio.create_task(self._auto_save_loop())
            logger.info("[跨会话记忆] 自动保存任务已启动")

    async def terminate(self):
        """插件终止"""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
        self._save_data()
        logger.info("[跨会话记忆] 插件已终止，数据已保存")

    def _get_group_for_session(self, session_id: str) -> Optional[MemoryGroup]:
        """获取会话所属的记忆组"""
        return self.session_to_group.get(session_id)

    async def _retrieve_relevant_memories(
        self,
        group: MemoryGroup,
        query_text: str,
        current_session_id: str
    ) -> List[MemoryRecord]:
        """检索相关的记忆"""
        if not self.use_embedding:
            # 简单模式：返回最近的记忆
            return [m for m in group.memories if m.session_id != current_session_id][-self.top_k:]

        # 嵌入模式：使用向量相似度检索
        query_embedding = await self._get_embedding(query_text)
        if not query_embedding:
            logger.warning("[跨会话记忆] 获取查询嵌入失败，回退到简单模式")
            return [m for m in group.memories if m.session_id != current_session_id][-self.top_k:]

        # 计算所有记忆的相似度
        scored_memories = []
        for memory in group.memories:
            if memory.session_id == current_session_id:
                continue

            if memory.embedding:
                similarity = self._calculate_similarity(query_embedding, memory.embedding)
                if similarity >= self.embedding_threshold:
                    scored_memories.append((similarity, memory))

        # 按相似度排序并返回 top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:self.top_k]]

    async def _format_memory_context(
        self,
        group: MemoryGroup,
        current_session_id: str,
        query_text: str
    ) -> List[dict]:
        """格式化记忆上下文"""
        # 检索相关记忆
        relevant_memories = await self._retrieve_relevant_memories(group, query_text, current_session_id)

        context = []
        for memory in relevant_memories:
            if self.include_sender_info:
                prefix = self.cross_group_prefix.format(
                    group_name=group.name,
                    sender=memory.sender_name or "用户",
                    time=memory.timestamp or ""
                )
                content = f"{prefix}{memory.content}"
            else:
                content = memory.content

            context.append({
                "role": memory.role,
                "content": content
            })

        return context

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求钩子 - 注入跨会话上下文"""
        if not self.enabled:
            return

        session_id = event.get_session_id()
        group = self._get_group_for_session(session_id)

        if not group:
            return

        # 获取当前消息文本
        query_text = event.message_str if hasattr(event, 'message_str') else ""

        # 获取相关记忆上下文
        memory_context = await self._format_memory_context(group, session_id, query_text)

        if memory_context:
            context_str = "\n".join([
                f"[{msg['role']}]: {msg['content']}"
                for msg in memory_context
            ])

            cross_memory_prompt = f"""
【跨群记忆上下文】
以下是来自其他群组的对话历史，你可以参考这些信息来更好地理解用户：
{context_str}
"""
            req.system_prompt = (req.system_prompt or "") + cross_memory_prompt
            logger.debug(f"[跨会话记忆] 已注入 {len(memory_context)} 条记忆")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息 - 记录用户消息"""
        if not self.enabled:
            return

        session_id = event.get_session_id()
        group = self._get_group_for_session(session_id)

        if not group:
            return

        # 记录用户消息
        content = event.message_str if hasattr(event, 'message_str') else ""
        if not content:
            return

        # 获取嵌入向量（如果启用）
        embedding = None
        if self.use_embedding:
            embedding = await self._get_embedding(content)

        record = MemoryRecord(
            id=f"{session_id}_{datetime.now().timestamp()}",
            role="user",
            content=content,
            session_id=session_id,
            group_name=group.name,
            sender_name=event.get_sender_name() or "用户",
            timestamp=datetime.now().strftime("%H:%M"),
            embedding=embedding
        )
        group.add_memory(record)
        logger.debug(f"[跨会话记忆] 记录用户消息: {session_id}")

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应钩子 - 记录助手回复"""
        if not self.enabled:
            return

        session_id = event.get_session_id()
        group = self._get_group_for_session(session_id)

        if not group:
            return

        # 记录助手回复
        if resp.result_chain:
            content = resp.result_chain.get_plain_text()
        else:
            content = resp.completion_text or ""

        if content:
            # 获取嵌入向量（如果启用）
            embedding = None
            if self.use_embedding:
                embedding = await self._get_embedding(content)

            record = MemoryRecord(
                id=f"{session_id}_bot_{datetime.now().timestamp()}",
                role="assistant",
                content=content,
                session_id=session_id,
                group_name=group.name,
                sender_name="Bot",
                timestamp=datetime.now().strftime("%H:%M"),
                embedding=embedding
            )
            group.add_memory(record)
            logger.debug(f"[跨会话记忆] 记录助手回复: {session_id}")

    @filter.command("memory_status")
    async def memory_status(self, event: AstrMessageEvent):
        """查看当前会话的记忆状态"""
        session_id = event.get_session_id()
        group = self._get_group_for_session(session_id)

        if not group:
            yield event.plain_result("当前会话未配置跨群记忆功能。")
            return

        status = f"""【跨会话记忆状态】
记忆组: {group.name}
组成员: {', '.join(group.session_ids)}
记忆数量: {len(group.memories)}
最大记录数: {group.max_history}
嵌入模式: {'启用' if self.use_embedding else '禁用'}
"""
        yield event.plain_result(status)

    @filter.command("memory_clear")
    async def memory_clear(self, event: AstrMessageEvent):
        """清除当前会话所属记忆组的历史"""
        session_id = event.get_session_id()
        group = self._get_group_for_session(session_id)

        if not group:
            yield event.plain_result("当前会话未配置跨群记忆功能。")
            return

        group.memories.clear()
        self._save_data()
        yield event.plain_result(f"已清除记忆组 [{group.name}] 的所有记忆记录。")

    @filter.command("memory_save")
    async def memory_save(self, event: AstrMessageEvent):
        """手动保存记忆数据"""
        self._save_data()
        yield event.plain_result("记忆数据已保存。")
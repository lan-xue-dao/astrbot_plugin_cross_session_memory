"""
AstrBot 跨会话记忆插件
让 Bot 能够跨群组/会话共享记忆，实现群组间的对话上下文互通
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
class MessageRecord:
    """消息记录"""
    role: str  # "user" 或 "assistant"
    content: str
    session_id: str
    sender_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MessageRecord":
        return cls(**data)


@dataclass
class MemoryGroup:
    """记忆组"""
    name: str
    session_ids: Set[str]
    max_history: int = 50
    history: List[MessageRecord] = field(default_factory=list)

    def add_message(self, record: MessageRecord):
        """添加消息记录"""
        self.history.append(record)
        # 保持历史记录在限制内
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "session_ids": list(self.session_ids),
            "max_history": self.max_history,
            "history": [r.to_dict() for r in self.history]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGroup":
        return cls(
            name=data["name"],
            session_ids=set(data.get("session_ids", [])),
            max_history=data.get("max_history", 50),
            history=[MessageRecord.from_dict(r) for r in data.get("history", [])]
        )


@register(
    "astrbot_plugin_cross_session_memory",
    "lanxuedao",
    "让 Bot 能够跨群组/会话共享记忆，实现群组间的对话上下文互通",
    "1.0.0",
    "https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory"
)
class CrossSessionMemoryPlugin(Star):
    """跨会话记忆插件"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 记忆组存储: session_id -> MemoryGroup
        self.session_to_group: Dict[str, MemoryGroup] = {}
        # 记忆组存储: group_name -> MemoryGroup
        self.memory_groups: Dict[str, MemoryGroup] = {}

        # 数据存储路径
        self.data_dir = os.path.join("data", "plugin_data", "astrbot_plugin_cross_session_memory")
        self.data_file = os.path.join(self.data_dir, "memory_data.json")

        # 自动保存任务
        self._save_task: Optional[asyncio.Task] = None
        self._running = True

        # 初始化
        self._init_config()
        self._load_data()

    def _init_config(self):
        """初始化配置"""
        # 获取配置值
        self.enabled = self.config.get("enabled", True)
        memory_groups_config = self.config.get("memory_groups", [])
        global_settings = self.config.get("global_settings", {})

        self.auto_save_interval = global_settings.get("auto_save_interval", 60)
        self.include_sender_info = global_settings.get("include_sender_info", True)
        self.cross_group_prefix = global_settings.get("cross_group_prefix", "[{group_name}] {sender}: ")

        # 初始化记忆组
        for group_config in memory_groups_config:
            group_name = group_config.get("group_name", "default")
            session_ids = group_config.get("session_ids", [])
            max_history = group_config.get("max_history", 50)

            if group_name in self.memory_groups:
                # 合并到现有组
                existing_group = self.memory_groups[group_name]
                existing_group.session_ids.update(session_ids)
                existing_group.max_history = max_history
            else:
                # 创建新组
                group = MemoryGroup(
                    name=group_name,
                    session_ids=set(session_ids),
                    max_history=max_history
                )
                self.memory_groups[group_name] = group

            # 建立映射
            for sid in session_ids:
                self.session_to_group[sid] = self.memory_groups[group_name]

        logger.info(f"[跨会话记忆] 已加载 {len(self.memory_groups)} 个记忆组")

    def _load_data(self):
        """从文件加载数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                memory_groups_data = data.get("memory_groups", {})
                # 确保 memory_groups_data 是字典类型
                if not isinstance(memory_groups_data, dict):
                    logger.warning(f"[跨会话记忆] memory_groups 数据格式错误，预期字典，实际类型: {type(memory_groups_data)}")
                    return

                for group_name, group_data in memory_groups_data.items():
                    if group_name in self.memory_groups:
                        # 更新现有组的历史记录
                        loaded_group = MemoryGroup.from_dict(group_data)
                        self.memory_groups[group_name].history = loaded_group.history

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

    def _format_cross_session_context(
        self,
        group: MemoryGroup,
        current_session_id: str
    ) -> List[dict]:
        """格式化跨会话上下文"""
        context = []

        for record in group.history:
            # 如果是当前会话的消息，保持原样
            if record.session_id == current_session_id:
                context.append({
                    "role": record.role,
                    "content": record.content
                })
            else:
                # 跨会话消息，添加前缀
                if self.include_sender_info:
                    prefix = self.cross_group_prefix.format(
                        group_name=group.name,
                        sender=record.sender_name or "用户",
                        time=record.timestamp or ""
                    )
                    content = f"{prefix}{record.content}"
                else:
                    content = record.content

                context.append({
                    "role": record.role,
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

        # 获取跨会话上下文
        cross_context = self._format_cross_session_context(group, session_id)

        if cross_context:
            # 将跨会话上下文添加到系统提示
            context_str = "\n".join([
                f"[{msg['role']}]: {msg['content']}"
                for msg in cross_context[-20:]  # 最近20条
            ])

            cross_memory_prompt = f"""
【跨群记忆上下文】
以下是来自其他群组的对话历史，你可以参考这些信息来更好地理解用户：
{context_str}
"""
            req.system_prompt = (req.system_prompt or "") + cross_memory_prompt
            logger.debug(f"[跨会话记忆] 已注入 {len(cross_context)} 条跨会话上下文")

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
        record = MessageRecord(
            role="user",
            content=event.message_str,
            session_id=session_id,
            sender_name=event.get_sender_name() or "用户",
            timestamp=datetime.now().strftime("%H:%M")
        )
        group.add_message(record)
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
            record = MessageRecord(
                role="assistant",
                content=content,
                session_id=session_id,
                sender_name="Bot",
                timestamp=datetime.now().strftime("%H:%M")
            )
            group.add_message(record)
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
历史记录数: {len(group.history)}
最大记录数: {group.max_history}
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

        group.history.clear()
        self._save_data()
        yield event.plain_result(f"已清除记忆组 [{group.name}] 的所有历史记录。")

    @filter.command("memory_save")
    async def memory_save(self, event: AstrMessageEvent):
        """手动保存记忆数据"""
        self._save_data()
        yield event.plain_result("记忆数据已保存。")

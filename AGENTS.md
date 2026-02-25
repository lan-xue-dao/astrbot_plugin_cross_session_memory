# AstrBot 跨会话记忆插件开发文档

## 项目概述

这是一个用于 AstrBot 的插件，实现了跨群组/会话的记忆共享功能。通过该插件，Bot 能够在不同群组之间共享对话上下文，实现群组间的对话历史互通。

## 技术栈

- 语言：Python 3.x
- 框架：AstrBot Plugin API
- 依赖：无额外依赖（使用 AstrBot 内置 API）

## 核心组件

### 数据模型

- MessageRecord：消息记录数据类
- MemoryGroup：记忆组数据类

### 插件类

- CrossSessionMemoryPlugin：主插件类，继承自 Star

### 事件钩子

- @filter.on_llm_request() - LLM 请求钩子
- @filter.event_message_type(filter.EventMessageType.ALL) - 消息监听
- @filter.on_llm_response() - LLM 响应钩子

### 管理命令

- memory_status - 查看当前会话的记忆状态
- memory_clear - 清除历史记录
- memory_save - 手动保存数据

## 开发规范

- 使用 Python 类型注解
- 使用 dataclass 定义数据模型
- 遵循 AstrBot 插件 API 规范
- 使用异步编程（async/await）

## GitHub

https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory
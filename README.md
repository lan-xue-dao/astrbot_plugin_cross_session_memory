# AstrBot 跨会话记忆插件

让 Bot 能够跨群组/会话共享记忆，实现群组间的对话上下文互通。

## 项目概述

这是一个用于 AstrBot 的插件，实现了跨群组/会话的记忆共享功能。通过该插件，Bot 能够在不同群组之间共享对话上下文，实现群组间的对话历史互通。

**主要特性：**
- 跨会话记忆共享：多个会话可以属于同一个记忆组，共享对话历史
- 自动持久化：记忆数据自动保存到本地文件，插件重启后可恢复
- 可配置的记忆组：支持创建多个记忆组，每组独立管理
- 灵活的消息格式：跨会话消息可自定义前缀格式
- 命令管理：提供命令行工具查看和清理记忆数据
- 自动保存循环：支持定时自动保存记忆数据

## 技术栈

- **语言：** Python 3.x
- **框架：** AstrBot Plugin API
- **依赖：** 无额外依赖（使用 AstrBot 内置 API）
- **核心库：** asyncio, json, dataclasses

## 安装

### 1. 获取插件

从 GitHub 仓库克隆或下载插件：

```bash
git clone https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory.git
```

### 2. 安装插件

将插件目录复制到 AstrBot 的插件目录中。

### 3. 配置插件

在 AstrBot 的插件配置界面中添加该插件的配置，按照配置说明设置记忆组。

### 4. 重启 AstrBot

重启 AstrBot 使插件生效。

### 5. 验证插件

在任意配置的会话中发送命令 `/memory_status` 查看插件状态。

## 配置说明

插件通过 `_conf_schema.json` 定义配置结构，主要配置项：

### 全局设置 (global_settings)

| 配置项 | 类型 | 默认值 | 约束 | 说明 |
|--------|------|--------|------|------|
| `enabled` | boolean | true | - | 是否启用跨会话记忆功能 |
| `auto_save_interval` | integer | 60 | 最小 10 | 自动保存间隔（秒） |
| `include_sender_info` | boolean | true | - | 是否在跨会话消息中包含发送者信息 |
| `cross_group_prefix` | string | `[{group_name}] {sender}: ` | - | 跨会话消息前缀格式，可用变量：`{group_name}`、`{sender}`、`{time}` |

### 记忆组配置 (memory_groups)

每个记忆组包含以下配置：

| 配置项 | 类型 | 必填 | 约束 | 说明 |
|--------|------|------|------|------|
| `group_name` | string | 是 | - | 记忆组名称 |
| `session_ids` | array | 是 | - | 该记忆组包含的会话ID列表（字符串数组） |
| `max_history` | integer | 否 | 1-500，默认 50 | 最大历史记录数 |

### 配置示例

```json
{
  "enabled": true,
  "global_settings": {
    "auto_save_interval": 60,
    "include_sender_info": true,
    "cross_group_prefix": "[{group_name}] {sender}: "
  },
  "memory_groups": [
    {
      "group_name": "我的好友群",
      "session_ids": ["group_123", "group_456", "private_789"],
      "max_history": 100
    }
  ]
}
```

## 命令参考

### `/memory_status`

查看当前会话的记忆状态。

**输出信息：**
- 记忆组名称
- 组成员（会话ID列表）
- 历史记录数
- 最大记录数

### `/memory_clear`

清除当前会话所属记忆组的所有历史记录。

**注意：** 此操作不可逆，将清除该记忆组的所有对话历史。

### `/memory_save`

手动保存记忆数据到文件。

**使用场景：** 当需要立即保存数据而不等待自动保存时使用。

## 工作原理

### 消息记录流程

1. 用户发送消息 → `on_message()` 钩子触发 → 记录到所属记忆组的 history
2. Bot 生成回复 → `on_llm_response()` 钩子触发 → 记录到所属记忆组的 history

### 上下文注入流程

1. LLM 请求发起 → `on_llm_request()` 钩子触发
2. 获取当前会话所属的记忆组
3. 从记忆组的 history 中提取历史消息
4. 格式化跨会话上下文（当前会话消息保持原样，其他会话消息添加前缀）
5. 将格式化的上下文注入到系统提示中

### 自动保存机制

1. 插件初始化时启动 `_auto_save_loop()` 异步任务
2. 按照配置的间隔自动调用 `_save_data()`
3. 插件终止时手动保存一次数据

## 数据存储

- **存储位置：** `data/plugin_data/astrbot_plugin_cross_session_memory/memory_data.json`
- **数据格式：** JSON 格式，包含所有记忆组的历史记录
- **持久化策略：** 定时自动保存（可配置间隔）+ 插件终止时保存 + 手动保存命令

## 项目结构

```
跨群聊天插件/
├── metadata.yaml       # 插件元数据（名称、版本、作者等）
├── main.py            # 插件主代码
├── requirements.txt   # 依赖声明
├── _conf_schema.json  # 配置架构定义
├── README.md          # 项目文档
├── AGENTS.md          # 开发者文档
└── .gitignore         # Git 忽略规则
```

## 核心组件

### 1. 数据模型

- **MessageRecord：** 消息记录数据类
  - `role`: 消息角色（"user" 或 "assistant"）
  - `content`: 消息内容
  - `session_id`: 会话ID
  - `sender_name`: 发送者名称
  - `timestamp`: 时间戳（格式：HH:MM）
  - 方法：`to_dict()`, `from_dict()`

- **MemoryGroup：** 记忆组数据类，管理一组会话及其历史消息
  - `name`: 记忆组名称
  - `session_ids`: 该组包含的会话ID集合
  - `max_history`: 最大历史记录数
  - `history`: 历史消息记录列表
  - 方法：`add_message()`, `to_dict()`, `from_dict()`

### 2. 插件类

- **CrossSessionMemoryPlugin：** 主插件类，继承自 `Star`
  - `__init__()`: 初始化配置、记忆组映射、数据存储路径
  - `initialize()`: 启动自动保存任务
  - `terminate()`: 停止插件并保存数据
  - `_init_config()`: 初始化配置，创建记忆组和会话映射
  - `_load_data()`: 从文件加载记忆数据
  - `_save_data()`: 保存数据到文件
  - `_auto_save_loop()`: 自动保存循环
  - `_get_group_for_session()`: 获取会话所属的记忆组
  - `_format_cross_session_context()`: 格式化跨会话上下文
  - 处理 LLM 请求和响应钩子
  - 提供管理命令

### 3. 事件钩子

- `@filter.on_llm_request()`：在 LLM 请求前注入跨会话上下文
- `@filter.event_message_type(filter.EventMessageType.ALL)`：监听所有用户消息并记录
- `@filter.on_llm_response()`：记录 Bot 的回复

### 4. 管理命令

- `memory_status`：查看当前会话的记忆状态
- `memory_clear`：清除当前会话所属记忆组的历史
- `memory_save`：手动保存记忆数据

## GitHub 仓库

- **仓库地址：** https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory
- **作者：** lanxuedao
- **版本：** 1.0.0
- **插件 ID：** astrbot_plugin_cross_session_memory

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0 (2026-02-25)
- 初始版本发布
- 实现跨会话记忆共享功能
- 支持多记忆组管理
- 自动持久化记忆数据
- 提供管理命令
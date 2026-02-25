# AstrBot 跨会话记忆插件

让 Bot 能够跨群组/会话共享记忆，实现群组间的对话上下文互通。

## 功能特性

- 跨会话记忆共享：多个会话可以属于同一个记忆组，共享对话历史
- 自动持久化：记忆数据自动保存到本地文件，插件重启后可恢复
- 可配置的记忆组：支持创建多个记忆组，每组独立管理
- 灵活的消息格式：跨会话消息可自定义前缀格式
- 命令管理：提供命令行工具查看和清理记忆数据

## 安装

1. 在 AstrBot 管理面板中搜索并安装插件
2. 或手动克隆仓库到 AstrBot 插件目录

## 配置

在 AstrBot 管理面板中配置插件参数：

### 全局设置

- **enabled**: 是否启用跨会话记忆功能（默认：true）
- **auto_save_interval**: 自动保存间隔（秒，默认：60）
- **include_sender_info**: 是否在跨会话消息中包含发送者信息（默认：true）
- **cross_group_prefix**: 跨会话消息的前缀格式（默认：[{group_name}] {sender}: ）

### 记忆组配置

- **group_name**: 记忆组名称
- **session_ids**: 该记忆组包含的会话ID列表
- **max_history**: 最大历史记录数（默认：50）

## 命令

- `/memory_status` - 查看当前会话的记忆状态
- `/memory_clear` - 清除当前会话所属记忆组的历史
- `/memory_save` - 手动保存记忆数据

## 工作原理

1. 用户发送消息时，插件自动记录到所属记忆组
2. Bot 生成回复时，插件记录到所属记忆组
3. 当用户在记忆组中的任意会话发起对话时，插件会注入该组的所有历史消息到 LLM 上下文中
4. 记忆数据定期自动保存，插件终止时也会保存

## 数据存储

数据存储位置：`data/plugin_data/astrbot_plugin_cross_session_memory/memory_data.json`

## GitHub

https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory

## 许可证

MIT
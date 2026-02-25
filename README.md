# AstrBot 跨会话记忆插件（基于向量嵌入）

让 Bot 能够跨群组/会话共享记忆，使用向量嵌入实现智能语义检索。

## 功能特性

- 🧠 **向量嵌入支持**：使用 AstrBot 内置的 Embedding Provider 进行语义向量存储
- 🔍 **智能检索**：基于向量相似度检索相关记忆，而非简单的时间顺序
- 📝 **自动记忆**：自动记录所有对话内容并生成嵌入向量
- 🔄 **跨会话共享**：多个会话可以属于同一个记忆组，共享记忆库
- 💾 **持久化存储**：记忆数据自动保存到本地文件，支持跨会话记忆
- ⚙️ **灵活配置**：支持简单模式和嵌入模式，可配置检索参数

## 工作模式

### 简单模式（默认）
- 按时间顺序返回最近的记忆
- 不需要配置 Embedding Provider
- 适合快速部署和测试

### 嵌入模式（推荐）
- 使用向量相似度进行语义检索
- 需要配置 AstrBot 的 Embedding Provider
- 智能返回最相关的记忆，效果更好

## 安装

1. 在 AstrBot 管理面板中搜索并安装插件
2. 或手动克隆仓库到 AstrBot 插件目录

## 配置

### 全局设置

- **enabled**: 是否启用跨会话记忆功能（默认：true）
- **auto_save_interval**: 自动保存间隔（秒，默认：60）
- **include_sender_info**: 是否在跨会话消息中包含发送者信息（默认：true）
- **cross_group_prefix**: 跨会话消息的前缀格式（默认：[{group_name}] {sender}: ）
- **use_embedding**: 是否启用向量嵌入模式（默认：false）
- **embedding_provider**: 嵌入服务提供商名称（需在 AstrBot 中配置）
- **top_k**: 检索返回的记忆数量（默认：5）
- **embedding_threshold**: 嵌入相似度阈值（默认：0.3）

### 记忆组配置

- **group_name**: 记忆组名称
- **session_ids**: 该记忆组包含的会话ID列表
- **max_history**: 最大历史记录数（默认：50）

## 启用嵌入模式

1. 在 AstrBot 管理面板中配置 Embedding Provider
   - 支持兼容 OpenAI API 的嵌入服务
   - 推荐使用免费或低成本的嵌入模型，如 BAAI/bge-m3

2. 在插件配置中设置：
   - `use_embedding`: true
   - `embedding_provider`: 填写你的 Embedding Provider 名称
   - 调整 `top_k` 和 `embedding_threshold` 以获得最佳效果

## 命令

- `/memory_status` - 查看当前会话的记忆状态
- `/memory_clear` - 清除当前会话所属记忆组的历史
- `/memory_save` - 手动保存记忆数据

## 工作原理

### 简单模式
1. 用户发送消息 → 记录到记忆组
2. Bot 生成回复 → 记录到记忆组
3. 当用户在记忆组中的任意会话发起对话时，返回最近的记忆

### 嵌入模式
1. 用户发送消息 → 记录到记忆组 → 生成嵌入向量
2. Bot 生成回复 → 记录到记忆组 → 生成嵌入向量
3. 当用户发起对话时，将当前消息转换为向量，在记忆库中检索最相似的记忆
4. 使用余弦相似度计算，返回相关性超过阈值的记忆

## 技术特点

- 使用余弦相似度计算向量相似度
- 支持增量更新，新对话自动添加到记忆库
- 自动持久化，重启后记忆保留
- 跨会话共享，实现真正的长期记忆

## 数据存储

数据存储位置：`data/plugin_data/astrbot_plugin_cross_session_memory/memory_data.json`

## GitHub

https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory

## 许可证

MIT
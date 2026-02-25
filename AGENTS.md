# AstrBot 跨会话记忆插件开发文档

## 项目概述

这是一个用于 AstrBot 的插件，实现了跨群组/会话的记忆共享功能。使用向量嵌入技术实现智能语义检索，比传统的时间顺序记忆更加智能。

## 技术栈

- 语言：Python 3.x
- 框架：AstrBot Plugin API
- 嵌入服务：AstrBot 内置 Embedding Provider
- 向量计算：余弦相似度

## 核心组件

### 数据模型

- MemoryRecord：记忆记录数据类
  - id: 唯一标识符
  - content: 记忆内容
  - session_id: 会话ID
  - group_name: 记忆组名称
  - sender_name: 发送者名称
  - role: 角色（user/assistant）
  - timestamp: 时间戳
  - embedding: 嵌入向量（可选）

- MemoryGroup：记忆组数据类
  - name: 记忆组名称
  - session_ids: 会话ID集合
  - max_history: 最大记录数
  - memories: 记忆记录列表

### 插件类

- CrossSessionMemoryPlugin：主插件类，继承自 Star
  - _init_config(): 初始化配置
  - _init_embedding(): 初始化嵌入服务
  - _get_embedding(): 获取文本的嵌入向量
  - _calculate_similarity(): 计算向量相似度
  - _retrieve_relevant_memories(): 检索相关记忆
  - _format_memory_context(): 格式化记忆上下文

### 事件钩子

- @filter.on_llm_request() - LLM 请求钩子
- @filter.event_message_type(filter.EventMessageType.ALL) - 消息监听
- @filter.on_llm_response() - LLM 响应钩子

### 管理命令

- memory_status - 查看状态
- memory_clear - 清除历史
- memory_save - 手动保存

## 开发规范

- 使用 Python 类型注解
- 使用 dataclass 定义数据模型
- 遵循 AstrBot 插件 API 规范
- 使用异步编程（async/await）

## 向量检索算法

1. 将查询文本转换为嵌入向量
2. 遍历记忆库中所有记忆的嵌入向量
3. 计算查询向量与每个记忆向量的余弦相似度
4. 过滤相似度低于阈值的记忆
5. 按相似度排序，返回 top_k 条记忆

## GitHub

https://github.com/lan-xue-dao/astrbot_plugin_cross_session_memory
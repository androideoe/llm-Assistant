# Ollama 多模型聊天助手

这是一个基于Ollama和LangChain的智能聊天助手，支持多模型切换、RAG（检索增强生成）功能、思考过程显示等高级特性，通过关联本地知识库进行智能问答。
![preview.gif](https://github.com/androideoe/llm-Assistant/blob/main/preview.gif)<br/>

## ✨ 功能特性

### 🤖 多模型支持
- **Llama 3.2**: 强大的开源语言模型
- **Qwen3 1.7B**: 支持思考过程显示的轻量级模型
- **动态切换**: 无需重启即可切换模型
- **模型状态**: 实时显示当前使用的模型

### 🧠 思考过程可视化
- **流式思考**: 实时显示Qwen3模型的思考过程
- **智能解析**: 自动识别`<think>`标签和思考内容
- **交互控制**: 支持展开/折叠思考过程
- **自动折叠**: 思考完成后可自动折叠显示

### 🔍 RAG 知识库功能
- **自动扫描**: 自动扫描本地知识库目录
- **智能检索**: 基于相似度的文档检索
- **相关性控制**: 可调节相关性阈值（0-1）
- **性能优化**: 缓存机制和自适应检索
- **多格式支持**: `.txt`、`.pdf`、`.md` 等格式

### 💬 会话管理
- **多会话支持**: 支持创建和管理多个对话会话
- **智能标题**: 根据对话内容自动生成会话标题
- **历史保存**: 本地存储对话历史
- **响应式UI**: 自适应不同屏幕尺寸

### 🚀 高级功能
- **流式响应**: 实时流式输出，提供流畅的用户体验
- **性能监控**: 内置性能分析和优化建议
- **健康检查**: API服务状态实时监控
- **错误处理**: 完善的错误处理和用户提示
- **缓存机制**: 智能缓存提升响应速度

## 📦 安装和部署

### 1. 环境准备
```bash
# 确保Python版本 >= 3.8
python --version

# 克隆项目
git clone <repository-url>
cd llm-Assistant
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 安装和配置Ollama
```bash
# 安装Ollama (根据你的操作系统)
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull llama3.2
ollama pull qwen3:1.7b
```

### 4. 准备知识库
在项目根目录创建知识库目录：
```bash
mkdir knowledge_base
```

将你的文档放入知识库目录：
```
llm-Assistant/
├── knowledge_base/          # 📁 知识库目录
│   ├── technical_docs/
│   │   ├── api_guide.md
│   │   └── architecture.pdf
│   ├── user_manual.txt
│   ├── faq.md
│   └── ...
├── main.py
├── index.html
└── requirements.txt
```

### 5. 启动服务
```bash
python main.py
```

服务启动后会显示：
- API服务地址: `http://localhost:8000`
- 知识库加载状态
- 可用模型列表

### 6. 访问前端
在浏览器中打开 `index.html` 文件，或访问 `http://localhost:8000/` （如果配置了静态文件服务）

## 🎯 使用指南

### 基本对话
1. **新建会话**: 点击"新建会话"按钮
2. **输入消息**: 在输入框中输入你的问题
3. **发送消息**: 点击发送按钮或按 `Enter` 键
4. **查看回复**: AI助手会实时流式回复

### 模型切换
1. **选择模型**: 在右上角模型选择器中选择目标模型
2. **点击切换**: 点击"切换"按钮
3. **等待加载**: 等待模型切换完成
4. **开始对话**: 新的对话将使用选定的模型

### RAG 知识库问答
1. **检查状态**: 点击"知识库状态"查看加载情况
2. **启用RAG**: 确保"使用RAG"选项已勾选
3. **调整阈值**: 根据需要调整相关性阈值
   - 阈值越高，检索越精确但可能遗漏相关内容
   - 阈值越低，检索越宽泛但可能包含不相关内容
4. **提问**: 输入关于知识库内容的问题

### 思考过程查看（Qwen3模型）
1. **选择Qwen3**: 切换到Qwen3 1.7B模型
2. **启用显示**: 确保"显示思考过程"已勾选
3. **观察思考**: 模型回答时会显示思考过程
4. **交互控制**: 点击思考过程标题可展开/折叠

## 🔧 配置说明

### 模型配置
```python
# 在main.py中可配置的参数
AVAILABLE_MODELS = ["llama3.2", "qwen3:1.7b"]
DEFAULT_MODEL = "qwen3:1.7b"
```

### 知识库配置
```python
# RAG相关配置
CHUNK_SIZE = 1000           # 文档分块大小
CHUNK_OVERLAP = 200         # 分块重叠长度  
MAX_DOCS = 5               # 最大检索文档数
MIN_DOCS = 1               # 最小检索文档数
DEFAULT_THRESHOLD = 0.5     # 默认相关性阈值
```

### 性能优化配置
```python
# 缓存配置
CACHE_TTL = 300            # 缓存生存时间（秒）
PERFORMANCE_MONITOR = True  # 是否启用性能监控
```

## 📊 API 接口文档

### 健康检查
```http
GET /api/health
```
返回API服务和模型状态

### 模型管理
```http
GET /api/models
POST /api/models/switch
```
获取可用模型列表和切换模型

### 会话管理
```http
POST /api/session/new
GET /api/session/history?session_id={id}
GET /api/session/documents?session_id={id}
```

### 聊天功能
```http
POST /api/chat/rag           # RAG对话
POST /api/chat/rag/stream    # RAG流式对话
POST /api/chat               # 普通对话
POST /api/chat/stream        # 普通流式对话
```

### 知识库管理
```http
GET /api/knowledge-base/status
POST /api/rag/debug
```

### 性能分析
```http
POST /api/performance/analyze
```

## 🏗️ 技术架构

### 后端架构
```
FastAPI Server
├── 模型管理 (Model Management)
│   ├── Ollama Client
│   ├── Model Switching
│   └── Health Monitoring
├── RAG系统 (RAG System)
│   ├── Document Loader
│   ├── Vector Store (FAISS)
│   ├── Retrieval Chain
│   └── Performance Monitor
├── 会话管理 (Session Management)
│   ├── Memory Management
│   ├── History Storage
│   └── Context Tracking
└── API层 (API Layer)
    ├── RESTful APIs
    ├── Streaming Support
    └── Error Handling
```

### 前端架构
```
Modern Web UI
├── 响应式布局 (Responsive Layout)
│   ├── Tailwind CSS
│   ├── 自适应设计
│   └── 移动端优化
├── 实时交互 (Real-time Interaction)
│   ├── Server-Sent Events
│   ├── 流式渲染
│   └── 思考过程显示
├── 状态管理 (State Management)
│   ├── 会话管理
│   ├── 本地存储
│   └── UI状态同步
└── 用户体验 (User Experience)
    ├── 智能标题生成
    ├── 加载动画
    └── 错误提示
```

## 📈 性能特性

### 缓存机制
- **结果缓存**: 检索结果缓存，避免重复计算
- **TTL控制**: 可配置的缓存生存时间
- **内存优化**: 智能内存管理和清理

### 自适应检索
- **动态调整**: 根据查询复杂度调整检索策略
- **相关性优化**: 多种相似度计算方法
- **性能监控**: 实时性能数据收集和分析

### 流式优化
- **实时渲染**: 内容实时流式渲染
- **思考过程**: 支持思考过程的流式显示
- **错误恢复**: 流式传输中断自动恢复

## 🐛 故障排除

### 常见问题

**1. Ollama连接失败**
```bash
# 检查Ollama服务状态
ollama list

# 重启Ollama服务
ollama serve
```

**2. 模型下载问题**
```bash
# 手动拉取模型
ollama pull llama3.2
ollama pull qwen3:1.7b

# 检查模型状态
ollama list
```

**3. 知识库加载失败**
- 检查 `knowledge_base` 目录权限
- 确保文档格式受支持
- 查看控制台日志获取详细错误信息

**4. 前端连接问题**
- 确认API服务运行在 `http://localhost:8000`
- 检查浏览器控制台是否有CORS错误
- 验证防火墙设置

**5. 思考过程不显示**
- 确保使用的是Qwen3模型
- 检查"显示思考过程"选项是否启用
- 验证模型输出是否包含`<think>`标签

### 日志调试
```bash
# 查看API日志
tail -f api.log

# 启用调试模式
export DEBUG=true
python main.py
```

## 📁 项目结构
```
llm-Assistant/
├── main.py                 # FastAPI后端主文件
├── index.html             # 前端页面
├── requirements.txt       # Python依赖列表
├── README.md             # 项目文档
├── knowledge_base/       # 知识库目录
│   ├── *.txt
│   ├── *.pdf  
│   ├── *.md
│   └── subdirectories/
├── vector_stores/        # 向量存储目录
│   ├── knowledge_base.faiss
│   ├── knowledge_base.pkl
│   └── metadata.json
├── api.log              # API运行日志
└── .gitignore           # Git忽略文件
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档改进
- 🎨 UI/UX优化
- ⚡ 性能优化
- 🧪 测试覆盖

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Ollama](https://ollama.com/) - 本地LLM运行环境
- [LangChain](https://langchain.com/) - LLM应用开发框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [Tailwind CSS](https://tailwindcss.com/) - 实用优先的CSS框架
- [FAISS](https://faiss.ai/) - 高效向量相似性搜索

## 📞 支持

如果你在使用过程中遇到问题：

1. 查看 [故障排除](#-故障排除) 部分
2. 搜索现有的 [Issues](../../issues)
3. 创建新的 Issue 描述你的问题
4. 加入我们的社区讨论

---

⭐ 如果这个项目对你有帮助，请给我们一个星标！
欢迎提交Issue和Pull Request来改进这个项目！

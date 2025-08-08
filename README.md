# Llama3.2 RAG 聊天助手

这是一个基于Ollama和LangChain的智能聊天助手，支持RAG（检索增强生成）功能，通过关联本地知识库进行智能问答。
![image](https://github.com/androideoe/llm-Assistant/blob/main/screenshot.png)

## 功能特性

### 🚀 核心功能
- **智能对话**: 基于Llama3.2模型的自然语言对话
- **知识库RAG**: 自动扫描本地知识库目录，支持多种文档格式
- **流式响应**: 实时流式输出，提供更好的用户体验
- **会话管理**: 多会话支持，历史对话保存
- **相关性控制**: 可调节相关性阈值，精确控制检索结果

### 📁 支持的知识库文件格式
- `.txt` - 文本文件
- `.pdf` - PDF文档
- `.md` - Markdown文件

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备知识库
在项目根目录创建 `knowledge_base` 文件夹，并将你的文档放入其中：
```
llm-Assistant/
├── knowledge_base/          # 📁 知识库目录
│   ├── document1.pdf
│   ├── document2.txt
│   ├── document3.md
│   └── subfolder/
│       └── document4.pdf
├── main.py
├── index.html
└── ...
```

### 3. 启动Ollama服务
确保已安装Ollama并下载llama3.2模型：
```bash
# 安装Ollama (如果还没安装)
# 下载llama3.2模型
ollama pull llama3.2
```

### 4. 启动API服务
```bash
python main.py
```
服务将在 `http://localhost:8000` 启动，并自动扫描知识库目录

### 5. 打开前端页面
在浏览器中打开 `index.html` 文件

## 使用指南

### 普通对话模式
1. 点击"新建会话"创建新对话
2. 在输入框中输入问题
3. 点击"发送"或按Enter键发送消息
4. 查看AI助手的回复

### RAG模式（知识库问答）
1. 确保知识库已加载（点击"知识库状态"查看）
2. 点击"RAG模式"切换到知识库问答模式
3. 输入关于知识库内容的问题
4. AI将基于知识库内容回答你的问题
5. 调节相关性阈值控制检索精度

### 会话管理
- **新建会话**: 点击"新建会话"按钮
- **切换会话**: 点击左侧会话列表中的会话
- **删除会话**: 点击会话项右侧的删除图标
- **会话历史**: 所有对话历史会自动保存

## 知识库管理

### 添加文档
1. 将文档文件放入 `knowledge_base` 目录
2. 重启应用，系统会自动扫描并处理新文档
3. 点击"知识库状态"查看加载状态

### 支持的文件结构
- 支持子目录嵌套
- 自动递归扫描所有支持的文件
- 支持中文文件名

### 向量索引
- 文档会被自动分割成小块
- 创建向量索引存储在 `vector_stores` 目录
- 重启应用时会自动加载已保存的索引

## API接口

### 健康检查
```
GET /api/health
```

### 会话管理
```
POST /api/session/new
GET /api/session/history?session_id={session_id}
```

### 普通对话
```
POST /api/chat
POST /api/chat/stream
```

### RAG功能
```
GET /api/knowledge-base/status
POST /api/chat/rag
POST /api/chat/rag/stream
```

## 技术架构

### 后端技术栈
- **FastAPI**: 高性能Web框架
- **LangChain**: LLM应用开发框架
- **Ollama**: 本地LLM服务
- **FAISS**: 向量数据库
- **SSE**: 服务器发送事件（流式响应）

### 前端技术栈
- **HTML5 + CSS3**: 现代化UI设计
- **Tailwind CSS**: 实用优先的CSS框架
- **JavaScript**: 原生JS，无框架依赖
- **Font Awesome**: 图标库

## 配置说明

### 模型配置
- 默认使用 `llama3.2` 模型
- 支持自定义系统提示词
- 可配置流式响应参数

### 知识库处理配置
- 文档分块大小: 1000字符
- 分块重叠: 200字符
- 检索文档数量: 2个
- 相关性阈值: 0.7（可调节）

## 故障排除

### 常见问题

1. **Ollama连接失败**
   - 确保Ollama服务正在运行
   - 检查是否已下载llama3.2模型
   - 验证端口8000未被占用

2. **知识库未加载**
   - 检查 `knowledge_base` 目录是否存在
   - 确保目录中有支持的文档文件
   - 查看应用日志了解具体错误

3. **RAG模式无响应**
   - 确保知识库已成功加载
   - 检查知识库状态按钮显示
   - 尝试降低相关性阈值

### 日志查看
应用运行时会生成 `api.log` 文件，包含详细的运行日志。

## 开发说明

### 项目结构
```
llm-Assistant/
├── main.py                  # FastAPI后端服务
├── index.html               # 前端页面
├── requirements.txt         # Python依赖
├── README.md               # 项目说明
├── knowledge_base/          # 知识库目录
│   └── *.txt, *.pdf, *.md
├── vector_stores/          # 向量索引存储
│   └── knowledge_base.faiss
└── api.log                 # 运行日志（自动生成）
```

### 扩展功能
- 支持更多文档格式（DOCX、PPT等）
- 添加文档预览功能
- 支持知识库热更新
- 添加用户认证和权限管理
- 支持多语言界面

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

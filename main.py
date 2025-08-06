from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import asyncio
import json
from typing import Optional
from langchain.memory import ConversationBufferMemory
import uuid
from fastapi import Query
from typing import Dict
from langchain.chains import ConversationChain

# 配置日志 - 更详细的日志设置
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="Llama3.2 API", description="与本地部署的Llama3.2模型交互的API接口", version="1.0")

# 配置CORS - 在生产环境中应指定具体的前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Ollama LLM - 更健壮的初始化
llm: Optional[Ollama] = None

# 会话内存池：session_id -> ConversationBufferMemory
session_memories: Dict[str, ConversationBufferMemory] = {}

def init_ollama():
    """初始化Ollama连接，带重试机制"""
    global llm
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            llm = Ollama(model="llama3.2")
            # 测试连接
            test_response = llm.invoke("ping")
            logger.info(f"成功连接到本地Ollama llama3.2模型，测试响应: {test_response[:30]}...")
            return True
        except Exception as e:
            logger.warning(f"连接到Ollama模型失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
    
    logger.error("无法初始化Ollama模型连接")
    return False

# 初始化模型
if not init_ollama():
    logger.warning("Ollama模型初始化失败，API仍将启动但部分功能可能无法使用")

# 定义请求体模型
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "你是一个 helpful 的助手。请用简洁明了的语言回答用户的问题。"
    stream: bool = False  # 是否启用流式响应
    session_id: Optional[str] = None  # 新增：会话ID

# 定义响应体模型
class ChatResponse(BaseModel):
    response: str
    model: str = "llama3.2"
    success: bool = True
    error: Optional[str] = None

# 新建会话API
@app.post("/api/session/new")
async def new_session():
    session_id = str(uuid.uuid4())
    session_memories[session_id] = ConversationBufferMemory(return_messages=True)
    return {"session_id": session_id}

# 获取会话历史API
@app.get("/api/session/history")
async def get_session_history(session_id: str = Query(...)):
    memory = session_memories.get(session_id)
    if not memory:
        return {"history": []}
    # 返回历史消息（user/ai分开）
    return {"history": [
        {"type": m.type, "content": m.content} for m in memory.chat_memory.messages
    ]}

@app.post("/api/chat", response_model=ChatResponse, description="与Llama3.2模型进行对话")
async def chat(request: ChatRequest):
    """普通对话接口，一次性返回完整响应"""
    try:
        # 检查模型是否初始化
        if llm is None:
            raise HTTPException(status_code=503, detail="模型未初始化，请检查Ollama服务")
        
        if request.stream:
            # 如果请求流式响应，引导到流式接口
            raise HTTPException(status_code=400, detail="请使用/api/chat/stream接口获取流式响应")
            
        logger.info(f"收到用户消息: {request.message[:100]}...")  # 限制日志长度
        
        # 获取/创建memory
        if not request.session_id:
            raise HTTPException(status_code=400, detail="缺少session_id")
        memory = session_memories.get(request.session_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            session_memories[request.session_id] = memory
        
        # 构造prompt，带历史
        prompt = PromptTemplate(
            input_variables=["input"],
            template=f"{request.system_prompt}\n\n{{history}}\n用户问：{{input}}\n助手答："
        )
        
        # 创建LLM链
        chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)
        
        # 运行链获取结果
        result = chain.run(request.message)
        
        # 验证结果
        if not result or not result.strip():
            logger.warning("模型返回空响应")
            return ChatResponse(
                response="抱歉，未能获取到有效响应。",
                success=False,
                error="模型返回空响应"
            )
        
        logger.info(f"模型响应: {result[:100]}...")  # 只打印前100个字符
        # 更新memory
        memory.chat_memory.add_user_message(request.message)
        memory.chat_memory.add_ai_message(result)
        return ChatResponse(response=result)
    
    except HTTPException:
        # 重新抛出FastAPI HTTP异常
        raise
    except Exception as e:
        # 捕获所有其他异常并记录
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        # 返回结构化错误响应
        return ChatResponse(
            response="",
            success=False,
            error=f"处理请求时出错: {str(e)}"
        )

@app.post("/api/chat/stream", description="与Llama3.2模型进行对话（流式响应）")
async def chat_stream(request: ChatRequest):
    """流式对话接口，逐段返回响应内容"""
    try:
        # 检查模型是否初始化
        if llm is None:
            raise HTTPException(status_code=503, detail="模型未初始化，请检查Ollama服务")
        
        logger.info(f"收到用户流式消息: {request.message[:100]}...")
        
        # 获取/创建memory
        if not request.session_id:
            raise HTTPException(status_code=400, detail="缺少session_id")
        memory = session_memories.get(request.session_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            session_memories[request.session_id] = memory
        
        # 拼接历史
        history = ""
        for m in memory.chat_memory.messages:
            if m.type == "human":
                history += f"用户：{m.content}\n"
            elif m.type == "ai":
                history += f"助手：{m.content}\n"
        
        # 创建提示模板
        prompt = f"{request.system_prompt}\n\n"
        # ConversationChain自动注入历史
        full_prompt = prompt + "{history}\n用户问：{input}\n助手答："
        prompt_filled = full_prompt.replace("{history}", history).replace("{input}", request.message)
        
        async def event_generator():
            try:
                # 使用Ollama的流式生成
                chunk_count = 0
                empty_chunks = 0
                full_response = ""
                
                for chunk in llm.stream(prompt_filled):
                    if chunk:
                        chunk_count += 1
                        empty_chunks = 0  # 重置空块计数器
                        full_response += chunk
                        yield ServerSentEvent(data=chunk, event="message")
                        await asyncio.sleep(0.01)  # 短暂延迟，防止前端处理不过来
                    else:
                        empty_chunks += 1
                        logger.warning(f"收到空的流式响应块 ({empty_chunks})")
                        if empty_chunks > 10:  # 连续10个空块则终止
                            logger.error("连续收到多个空块，终止流式响应")
                            yield ServerSentEvent(data="", event="error", id="empty_chunks")
                            break
                
                if chunk_count == 0:
                    logger.warning("未收到任何有效流式响应块")
                    yield ServerSentEvent(data="未收到有效响应", event="error", id="no_content")
                else:
                    # 发送结束信号
                    yield ServerSentEvent(data="", event="end")
                    
                # 更新memory
                memory.chat_memory.add_user_message(request.message)
                memory.chat_memory.add_ai_message(full_response)
                    
            except Exception as e:
                logger.error(f"流式处理出错: {str(e)}", exc_info=True)
                yield ServerSentEvent(data=f"处理出错: {str(e)}", event="error")
        
        return EventSourceResponse(event_generator())
    
    except HTTPException:
        # 重新抛出FastAPI HTTP异常
        raise
    except Exception as e:
        logger.error(f"处理流式请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理流式请求时出错: {str(e)}")

@app.get("/api/health", description="检查API和模型健康状态")
async def health_check():
    """检查API和模型是否正常运行"""
    try:
        # 检查模型是否初始化
        if llm is None:
            return {
                "status": "unhealthy",
                "model": "llama3.2",
                "message": "模型未初始化，请检查Ollama服务",
                "test_response": None
            }
        
        # 简单测试模型是否可用
        test_response = llm.invoke("你好")
        return {
            "status": "healthy",
            "model": "llama3.2",
            "message": "API和模型均正常运行",
            "test_response": test_response[:50]  # 返回部分测试响应
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "model": "llama3.2",
            "message": f"模型不可用: {str(e)}",
            "test_response": None
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动API服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import asyncio
from typing import Optional
from langchain.memory import ConversationBufferMemory
import uuid
from fastapi import Query
from typing import Dict
from langchain.chains import ConversationChain
# RAG相关导入
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import os
import time
from pathlib import Path
import functools
from typing import Tuple, List, Optional
import time

# 添加缓存机制
# 缓存装饰器
def cache_result(ttl_seconds: int = 300):
    """简单的内存缓存装饰器"""
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # 检查缓存
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

# 添加性能监控
import time
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name: str):
    """性能监控上下文管理器"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"性能监控 - {operation_name}: {elapsed:.3f}秒")

# 优化后的相似度计算函数
def calculate_similarity_score_optimized(distance_score: float, method: str = "sigmoid") -> float:
    """
    优化后的相似度计算函数，减少复杂计算
    """
    import math
    
    # 处理异常值
    if distance_score < 0:
        distance_score = 0
    
    if method == "cosine":
        # 对于归一化向量，直接使用
        return max(0, min(1, distance_score))
    elif method == "sigmoid":
        # 简化的sigmoid计算
        try:
            if distance_score > 1000:
                return 1 / (1 + math.log(1 + distance_score / 1000))
            elif distance_score > 100:
                scaled_distance = distance_score / 100
                exp_term = min(scaled_distance - 2, 700)
                exp_term = max(exp_term, -700)
                return 1 / (1 + math.exp(exp_term))
            else:
                # 简化的sigmoid计算
                exp_term = min(distance_score - 10, 700)
                exp_term = max(exp_term, -700)
                similarity = 1 / (1 + math.exp(exp_term))
                return min(similarity, 0.95)
        except (OverflowError, ValueError):
            return max(0, 1 - distance_score / 10000.0)
    else:
        # 默认线性转换
        normalized_distance = min(distance_score / 1000.0, 1.0)
        return max(0, 1 - normalized_distance)

# 优化后的检索函数（带性能监控）
@cache_result(ttl_seconds=60)  # 缓存1分钟
def optimized_retrieval(vector_store, query: str, max_docs: int = 5, 
                       relevance_threshold: float = 0.5) -> Tuple[List, float, Dict]:
    """
    优化后的检索函数，减少不必要的计算和日志
    """
    retrieval_start_time = time.time()
    logger.info(f"🔍 RAG检索开始 - 查询: '{query[:50]}...' | 最大文档数: {max_docs} | 相关性阈值: {relevance_threshold}")
    
    with performance_monitor("向量检索"):
        try:
            # 直接检索所需数量的文档
            initial_k = min(max_docs * 2, 10)
            logger.debug(f"📊 初始检索参数 - K值: {initial_k}")
            
            # 执行检索
            search_start_time = time.time()
            docs = vector_store.similarity_search_with_score(query, k=initial_k)
            search_duration = time.time() - search_start_time
            logger.info(f"⚡ 向量检索完成 - 耗时: {search_duration:.3f}秒 | 原始结果数: {len(docs)}")
            
            if not docs:
                logger.warning("❌ RAG检索失败 - 未找到任何相关文档")
                return [], 0.0, {"retrieved": 0, "relevant": 0, "threshold": relevance_threshold}
            
            # 快速处理结果
            doc_scores = []
            distances = []
            
            logger.debug("📈 开始处理检索结果和计算相似度分数...")
            for i, result in enumerate(docs):
                try:
                    if isinstance(result, tuple) and len(result) == 2:
                        doc, score = result
                        score_float = float(score)
                        distances.append(score_float)
                        logger.debug(f"  文档 {i+1}: 距离分数 = {score_float:.4f}")
                    else:
                        logger.warning(f"⚠️  文档 {i+1}: 格式异常，跳过")
                        continue
                except Exception as e:
                    logger.error(f"❌ 处理文档 {i+1} 时出错: {str(e)}")
                    continue
            
            if not distances:
                logger.error("❌ RAG检索失败 - 无有效距离分数")
                return [], 0.0, {"retrieved": len(docs), "relevant": 0, "threshold": relevance_threshold}
            
            # 距离统计信息
            min_distance = min(distances)
            max_distance = max(distances)
            avg_distance = sum(distances) / len(distances)
            logger.info(f"📊 距离分数统计 - 最小: {min_distance:.4f} | 最大: {max_distance:.4f} | 平均: {avg_distance:.4f}")
            
            # 快速检测向量类型
            is_normalized = max(distances) <= 1.0 and min(distances) >= 0.0
            similarity_method = "cosine" if is_normalized else "sigmoid"
            logger.debug(f"🔧 相似度计算方法: {similarity_method} (归一化: {is_normalized})")
            
            # 批量计算相似度
            similarity_start_time = time.time()
            for i, (doc, score_float) in enumerate(zip([result[0] for result in docs if isinstance(result, tuple) and len(result) == 2], distances)):
                try:
                    similarity_score = calculate_similarity_score_optimized(score_float, method=similarity_method)
                    doc_scores.append((doc, similarity_score))
                    logger.debug(f"  文档 {i+1}: 距离 {score_float:.4f} -> 相似度 {similarity_score:.4f}")
                except Exception as e:
                    logger.error(f"❌ 计算文档 {i+1} 相似度时出错: {str(e)}")
                    continue
            
            similarity_duration = time.time() - similarity_start_time
            logger.debug(f"⚡ 相似度计算完成 - 耗时: {similarity_duration:.3f}秒")
            
            # 排序并过滤
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            max_similarity = doc_scores[0][1] if doc_scores else 0.0
            logger.info(f"🏆 最高相似度分数: {max_similarity:.4f}")
            
            # 过滤相关文档
            relevant_docs = []
            filtered_scores = []
            for i, (doc, similarity_score) in enumerate(doc_scores):
                if similarity_score >= relevance_threshold:
                    relevant_docs.append(doc)
                    filtered_scores.append(similarity_score)
                    
                    # 记录文档内容预览
                    content_preview = doc.page_content[:100].replace('\n', ' ')
                    logger.info(f"✅ 相关文档 {len(relevant_docs)}: 相似度 {similarity_score:.4f} | 内容: '{content_preview}...'")
                    
                    if len(relevant_docs) >= max_docs:
                        logger.info(f"📋 达到最大文档数限制 ({max_docs})")
                        break
                else:
                    logger.debug(f"❌ 文档 {i+1}: 相似度 {similarity_score:.4f} < 阈值 {relevance_threshold}，已过滤")
            
            retrieval_info = {
                "retrieved": len(docs),
                "relevant": len(relevant_docs),
                "threshold": relevance_threshold,
                "max_similarity": max_similarity,
                "similarity_method": similarity_method,
                "distance_stats": {
                    "min": min_distance,
                    "max": max_distance,
                    "avg": avg_distance
                },
                "is_normalized": is_normalized,
                "relevant_scores": filtered_scores,
                "search_duration": search_duration,
                "similarity_duration": similarity_duration
            }
            
            total_duration = time.time() - retrieval_start_time
            
            if relevant_docs:
                logger.info(f"🎯 RAG检索成功 - 总耗时: {total_duration:.3f}秒 | 检索到: {len(docs)}个 | 相关: {len(relevant_docs)}个 | 平均相似度: {sum(filtered_scores)/len(filtered_scores):.4f}")
            else:
                logger.warning(f"⚠️  RAG检索完成但无相关文档 - 总耗时: {total_duration:.3f}秒 | 检索到: {len(docs)}个 | 最高相似度: {max_similarity:.4f} < 阈值: {relevance_threshold}")
            
            return relevant_docs, max_similarity, retrieval_info
            
        except Exception as e:
            total_duration = time.time() - retrieval_start_time
            logger.error(f"❌ RAG检索异常 - 总耗时: {total_duration:.3f}秒 | 错误: {str(e)}", exc_info=True)
            return [], 0.0, {"retrieved": 0, "relevant": 0, "threshold": relevance_threshold, "error": str(e)}

# 替换原有的adaptive_retrieval函数
def adaptive_retrieval(vector_store, query: str, max_docs: int = 5, min_docs: int = 1, 
                       relevance_threshold: float = 0.5) -> tuple:
    """
    使用优化后的检索函数
    """
    return optimized_retrieval(vector_store, query, max_docs, relevance_threshold)

# 设置工作目录为当前脚本所在目录
WORKING_DIR = Path(__file__).parent
os.chdir(WORKING_DIR)

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
app = FastAPI(title="多模型聊天 API", description="与本地部署的Ollama模型交互的API接口，默认使用Qwen3 1.7B（支持思考过程），可切换至Llama 3.2", version="1.0")

# 配置CORS - 在生产环境中应指定具体的前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Ollama LLM - 更健壮的初始化，支持多模型
llm: Optional[Ollama] = None
current_model: str = "qwen3:1.7b"  # 默认模型改为Qwen3
available_models = ["llama3.2", "qwen3:1.7b"]  # 可用模型列表

# 会话内存池：session_id -> ConversationBufferMemory
session_memories: Dict[str, ConversationBufferMemory] = {}

# RAG相关全局变量
embeddings: Optional[OllamaEmbeddings] = None
vector_store: Optional[FAISS] = None
document_sessions: Dict[str, FAISS] = {}  # session_id -> FAISS vector store

# 知识库目录配置 - 使用相对路径
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
VECTOR_STORES_DIR = Path("vector_stores")

# 确保存储目录存在
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
VECTOR_STORES_DIR.mkdir(exist_ok=True)

logger.info(f"工作目录设置为: {WORKING_DIR}")
logger.info(f"知识库目录: {KNOWLEDGE_BASE_DIR.absolute()}")
logger.info(f"向量存储目录: {VECTOR_STORES_DIR.absolute()}")

# 知识库文件类型
SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.md'}

def init_ollama(model_name: str = "qwen3:1.7b"):
    """初始化Ollama连接，带重试机制，支持指定模型"""
    global llm, current_model
    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            llm = Ollama(model=model_name)
            # 测试连接
            test_response = llm.invoke("ping")
            current_model = model_name
            logger.info(f"成功连接到本地Ollama {model_name}模型，测试响应: {test_response[:30]}...")
            return True
        except Exception as e:
            logger.warning(f"连接到Ollama {model_name}模型失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
    
    logger.error(f"无法初始化Ollama {model_name}模型连接")
    return False

# 初始化模型
if not init_ollama():
    logger.warning("Ollama模型初始化失败，API仍将启动但部分功能可能无法使用")

def init_rag(model_name: str = "qwen3:1.7b"):
    """初始化RAG相关组件，支持指定嵌入模型"""
    global embeddings
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        logger.info(f"成功初始化Ollama嵌入模型: {model_name}")
        return True
    except Exception as e:
        logger.error(f"初始化嵌入模型失败: {str(e)}")
        return False

# 初始化RAG
if not init_rag():
    logger.warning("RAG组件初始化失败，文档功能可能无法使用")

def get_knowledge_base_files():
    """获取知识库中的所有文件信息"""
    files_info = {}
    if not KNOWLEDGE_BASE_DIR.exists():
        return files_info
    
    for file_path in KNOWLEDGE_BASE_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files_info[str(file_path)] = {
                'mtime': file_path.stat().st_mtime,
                'size': file_path.stat().st_size
            }
    return files_info

def get_vector_store_info():
    """获取向量存储的信息"""
    vector_store_path = VECTOR_STORES_DIR / "knowledge_base.faiss"
    metadata_path = VECTOR_STORES_DIR / "knowledge_base_metadata.json"
    
    # 检查向量存储目录是否存在
    if not vector_store_path.exists() or not vector_store_path.is_dir():
        return None
    
    # 检查必要的FAISS文件是否存在
    index_faiss_path = vector_store_path / "index.faiss"
    index_pkl_path = vector_store_path / "index.pkl"
    
    if not index_faiss_path.exists() or not index_pkl_path.exists():
        return None
    
    try:
        import json
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 如果没有元数据文件，返回基本信息
            return {
                'exists': True,
                'mtime': vector_store_path.stat().st_mtime
            }
    except Exception as e:
        logger.warning(f"读取向量存储信息失败: {str(e)}")
        return None

def save_vector_store_metadata(files_info):
    """保存向量存储的元数据"""
    try:
        import json
        metadata_path = VECTOR_STORES_DIR / "knowledge_base_metadata.json"
        metadata = {
            'files_info': files_info,
            'created_at': time.time(),
            'document_count': sum(len(files_info) for files_info in [files_info])
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("保存向量存储元数据")
    except Exception as e:
        logger.warning(f"保存向量存储元数据失败: {str(e)}")

def needs_rebuild():
    """检查是否需要重新构建向量索引"""
    # 获取当前知识库文件信息
    current_files = get_knowledge_base_files()
    if not current_files:
        logger.info("知识库目录为空，无需构建向量索引")
        return False
    
    # 获取向量存储信息
    vector_info = get_vector_store_info()
    if not vector_info:
        logger.info("向量存储不存在，需要重新构建")
        return True
    
    # 检查是否有元数据
    if 'files_info' not in vector_info:
        logger.info("向量存储缺少元数据，需要重新构建")
        return True
    
    # 比较文件信息
    stored_files = vector_info['files_info']
    
    # 检查文件数量是否变化
    if len(current_files) != len(stored_files):
        logger.info(f"文件数量变化: 当前{len(current_files)}个，存储{len(stored_files)}个")
        return True
    
    # 检查文件是否有变化
    for file_path, current_info in current_files.items():
        if file_path not in stored_files:
            logger.info(f"新增文件: {file_path}")
            return True
        
        stored_info = stored_files[file_path]
        if (current_info['mtime'] != stored_info['mtime'] or 
            current_info['size'] != stored_info['size']):
            logger.info(f"文件已修改: {file_path}")
            return True
    
    # 检查是否有文件被删除
    for file_path in stored_files:
        if file_path not in current_files:
            logger.info(f"文件已删除: {file_path}")
            return True
    
    logger.info("知识库文件无变化，使用现有向量索引")
    return False

def load_saved_vector_stores():
    """加载已保存的向量索引"""
    global vector_store
    
    try:
        if not VECTOR_STORES_DIR.exists():
            return False
        
        # 优先加载归一化向量索引
        normalized_path = VECTOR_STORES_DIR / "knowledge_base_normalized.faiss"
        if normalized_path.exists() and normalized_path.is_dir():
            try:
                # 检查必要的文件是否存在
                index_faiss_path = normalized_path / "index.faiss"
                index_pkl_path = normalized_path / "index.pkl"
                
                if not index_faiss_path.exists() or not index_pkl_path.exists():
                    logger.warning(f"归一化向量索引文件不完整: {normalized_path}")
                    raise Exception("索引文件不完整")
                
                vector_store = FAISS.load_local(str(normalized_path), embeddings)
                logger.info(f"加载归一化向量索引: {normalized_path}")
                return True
            except Exception as e:
                logger.warning(f"加载归一化向量索引失败 {normalized_path}: {str(e)}")
        
        # 如果归一化索引不存在，尝试加载原始索引
        knowledge_base_path = VECTOR_STORES_DIR / "knowledge_base.faiss"
        if knowledge_base_path.exists() and knowledge_base_path.is_dir():
            try:
                vector_store = FAISS.load_local(str(knowledge_base_path), embeddings)
                logger.info(f"加载原始向量索引: {knowledge_base_path}")
                return True
            except Exception as e:
                logger.warning(f"加载原始向量索引失败 {knowledge_base_path}: {str(e)}")
                return False
        
        return False
                    
    except Exception as e:
        logger.error(f"加载向量索引时出错: {str(e)}")
        return False

def load_knowledge_base():
    """智能加载知识库 - 只在必要时重新处理文档"""
    try:
        import time
        
        print(f"开始智能加载知识库...")
        logger.info("开始智能加载知识库...")
        
        # 检查是否需要重新构建
        if needs_rebuild():
            print("检测到知识库变化，重新构建向量索引...")
            logger.info("检测到知识库变化，重新构建向量索引...")
            return rebuild_knowledge_base()
        else:
            # 尝试加载现有向量索引
            if load_saved_vector_stores():
                print("成功加载现有向量索引")
                logger.info("成功加载现有向量索引")
                return True
            else:
                print("加载现有向量索引失败，重新构建...")
                logger.info("加载现有向量索引失败，重新构建...")
                return rebuild_knowledge_base()
        
    except Exception as e:
        print(f"加载知识库失败: {str(e)}")
        logger.error(f"加载知识库失败: {str(e)}")
        return False

def rebuild_knowledge_base():
    """重新构建知识库向量索引"""
    try:
        import time
        build_start_time = time.time()
        
        logger.info("🔨 开始重新构建知识库向量索引...")
        print(f"开始扫描知识库目录: {KNOWLEDGE_BASE_DIR}")
        
        if not KNOWLEDGE_BASE_DIR.exists():
            logger.error(f"❌ 知识库目录不存在: {KNOWLEDGE_BASE_DIR}")
            print(f"知识库目录不存在: {KNOWLEDGE_BASE_DIR}")
            return False
        
        # 扫描所有支持的文件
        all_documents = []
        logger.info(f"📁 扫描支持的文件类型: {SUPPORTED_EXTENSIONS}")
        print(f"支持的扩展名: {SUPPORTED_EXTENSIONS}")
        
        for file_path in KNOWLEDGE_BASE_DIR.rglob("*"):
            print(f"检查文件: {file_path}, 是文件: {file_path.is_file()}, 扩展名: {file_path.suffix.lower()}")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_documents.append(file_path)
                file_size = file_path.stat().st_size
                logger.info(f"✅ 添加文档: {file_path.name} (大小: {file_size} bytes)")
                print(f"添加文档: {file_path}")
        
        logger.info(f"📊 扫描完成 - 找到 {len(all_documents)} 个文档文件")
        print(f"找到 {len(all_documents)} 个文档文件")
        
        if not all_documents:
            logger.warning("⚠️  知识库目录中没有找到支持的文档文件")
            print("知识库目录中没有找到支持的文档文件")
            return False
        
        # 处理所有文档
        all_splits = []
        processing_start_time = time.time()
        logger.info("📄 开始处理文档内容...")
        
        for i, file_path in enumerate(all_documents):
            try:
                file_start_time = time.time()
                logger.info(f"📖 处理文档 {i+1}/{len(all_documents)}: {file_path.name}")
                print(f"处理文档: {file_path}")
                
                # 根据文件类型选择加载器
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    logger.debug(f"🔧 使用PDF加载器: {file_path.name}")
                else:
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    logger.debug(f"🔧 使用文本加载器: {file_path.name}")
                
                # 加载文档
                documents = loader.load()
                doc_content_length = sum(len(doc.page_content) for doc in documents)
                logger.info(f"📋 文档加载成功: {file_path.name} | 文档数: {len(documents)} | 总长度: {doc_content_length}字符")
                print(f"加载文档成功: {file_path}, 文档数量: {len(documents)}")
                
                # 文本分割
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                splits = text_splitter.split_documents(documents)
                all_splits.extend(splits)
                
                file_duration = time.time() - file_start_time
                avg_chunk_size = sum(len(split.page_content) for split in splits) / len(splits) if splits else 0
                logger.info(f"✂️  文档分割完成: {file_path.name} | 分割数: {len(splits)} | 平均块大小: {avg_chunk_size:.0f}字符 | 耗时: {file_duration:.3f}秒")
                print(f"处理文档 {file_path.name}: {len(splits)} 个文档块")
                
            except Exception as e:
                logger.error(f"❌ 处理文档失败: {file_path} | 错误: {str(e)}", exc_info=True)
                print(f"处理文档 {file_path} 失败: {str(e)}")
                continue
        
        processing_duration = time.time() - processing_start_time
        
        if not all_splits:
            logger.error("❌ 没有成功处理任何文档")
            print("没有成功处理任何文档")
            return False
        
        total_content_length = sum(len(split.page_content) for split in all_splits)
        avg_chunk_size = total_content_length / len(all_splits)
        logger.info(f"📊 文档处理统计 - 总块数: {len(all_splits)} | 总长度: {total_content_length}字符 | 平均块大小: {avg_chunk_size:.0f}字符 | 处理耗时: {processing_duration:.3f}秒")
        print(f"总共处理了 {len(all_splits)} 个文档块")
        
        # 创建全局向量存储
        if embeddings is None:
            logger.error("❌ 嵌入模型未初始化")
            print("嵌入模型未初始化")
            return False
        
        logger.info("🧮 开始创建向量存储...")
        print("开始创建向量存储...")
        
        global vector_store
        vector_start_time = time.time()
        
        # 使用余弦相似度而不是L2距离
        vector_store = FAISS.from_documents(all_splits, embeddings, distance_strategy="COSINE")
        
        vector_duration = time.time() - vector_start_time
        logger.info(f"🎯 向量存储创建完成 - 耗时: {vector_duration:.3f}秒")
        
        # 保存向量索引
        save_start_time = time.time()
        vector_store_path = VECTOR_STORES_DIR / "knowledge_base.faiss"
        vector_store.save_local(str(vector_store_path))
        
        # 保存元数据
        files_info = get_knowledge_base_files()
        save_vector_store_metadata(files_info)
        
        save_duration = time.time() - save_start_time
        total_duration = time.time() - build_start_time
        
        logger.info(f"💾 向量索引保存完成 - 保存耗时: {save_duration:.3f}秒")
        logger.info(f"✅ 知识库构建成功 - 总耗时: {total_duration:.3f}秒 | 文档: {len(all_documents)}个 | 块数: {len(all_splits)}个 | 向量维度: {len(embeddings.embed_query('test'))}")
        
        print(f"成功创建知识库向量索引，包含 {len(all_splits)} 个文档块")
        return True
        
    except Exception as e:
        build_duration = time.time() - build_start_time if 'build_start_time' in locals() else 0
        logger.error(f"❌ 知识库构建失败 - 耗时: {build_duration:.3f}秒 | 错误: {str(e)}", exc_info=True)
        print(f"重新构建知识库失败: {str(e)}")
        return False

# 加载知识库
if not load_knowledge_base():
    logger.warning("知识库加载失败，RAG功能可能无法使用")

# 定义请求体模型
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "你是一个 helpful 的助手。请用简洁明了的语言回答用户的问题。"
    stream: bool = False  # 是否启用流式响应
    session_id: Optional[str] = None  # 新增：会话ID
    relevance_threshold: float = 0.5  # 提高默认阈值到0.5
    use_rag: bool = True  # 新增：是否使用RAG功能，默认为True
    max_docs: int = 5  # 新增：最大检索文档数量
    min_docs: int = 1  # 新增：最小检索文档数量

class ModelSwitchRequest(BaseModel):
    model_name: str

# 定义响应体模型
class ChatResponse(BaseModel):
    response: str
    model: str  # 移除默认值，动态设置
    success: bool = True
    error: Optional[str] = None

# 获取可用模型列表
@app.get("/api/models")
async def get_available_models():
    """获取可用的模型列表"""
    return {
        "available_models": available_models,
        "current_model": current_model
    }

# 切换模型
@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """切换当前使用的模型"""
    try:
        model_name = request.model_name
        
        if model_name not in available_models:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的模型: {model_name}，可用模型: {available_models}"
            )
        
        # 初始化新模型
        if not init_ollama(model_name):
            raise HTTPException(
                status_code=503, 
                detail=f"无法初始化模型: {model_name}"
            )
        
        # 初始化RAG组件（如果需要）
        if not init_rag(model_name):
            logger.warning(f"RAG组件初始化失败，模型 {model_name} 的RAG功能可能无法使用")
        
        logger.info(f"成功切换到模型: {model_name}")
        return {
            "success": True,
            "message": f"成功切换到模型: {model_name}",
            "current_model": current_model,
            "previous_model": model_name if model_name != current_model else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"切换模型失败: {str(e)}")

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

# 获取会话文档列表
@app.get("/api/session/documents")
async def get_session_documents(session_id: str = Query(...)):
    """获取会话的文档列表"""
    try:
        session_doc_dir = KNOWLEDGE_BASE_DIR / session_id
        if not session_doc_dir.exists():
            return {"documents": []}
        
        documents = []
        for doc_file in session_doc_dir.iterdir():
            if doc_file.is_file():
                documents.append({
                    "name": doc_file.name,
                    "size": doc_file.stat().st_size,
                    "upload_time": doc_file.stat().st_mtime
                })
        
        return {"documents": documents}
    except Exception as e:
        logger.error(f"获取会话文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

# 知识库状态接口
@app.get("/api/knowledge-base/status")
async def get_knowledge_base_status():
    """获取知识库状态"""
    try:
        if not KNOWLEDGE_BASE_DIR.exists():
            return {
                "status": "not_found",
                "message": "知识库目录不存在",
                "document_count": 0
            }
        
        # 统计文档数量
        document_count = 0
        for file_path in KNOWLEDGE_BASE_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                document_count += 1
        
        return {
            "status": "loaded" if vector_store is not None else "not_loaded",
            "message": f"知识库包含 {document_count} 个文档" if document_count > 0 else "知识库目录为空",
            "document_count": document_count
        }
    except Exception as e:
        logger.error(f"获取知识库状态失败: {str(e)}")
        return {
            "status": "error",
            "message": f"获取知识库状态失败: {str(e)}",
            "document_count": 0
        }

# RAG调试接口
@app.post("/api/rag/debug")
async def debug_rag_retrieval(
    query: str = Query(..., description="查询文本"),
    relevance_threshold: float = Query(0.5, description="相关性阈值"),
    max_docs: int = Query(5, description="最大检索文档数"),
    test_embedding: bool = Query(False, description="是否测试嵌入模型")
):
    """调试RAG检索过程，返回详细的检索信息"""
    debug_start_time = time.time()
    logger.info(f"🔬 RAG调试请求开始 - 查询: '{query[:50]}...' | 阈值: {relevance_threshold} | 最大文档数: {max_docs} | 测试嵌入: {test_embedding}")
    
    try:
        if vector_store is None:
            logger.error("❌ 调试失败: 知识库未加载")
            raise HTTPException(status_code=503, detail="知识库未加载")
        
        if embeddings is None:
            logger.error("❌ 调试失败: 嵌入模型未初始化")
            raise HTTPException(status_code=503, detail="嵌入模型未初始化")
        
        # 如果请求测试嵌入模型
        if test_embedding:
            logger.info("🧪 开始测试嵌入模型...")
            try:
                # 测试嵌入模型
                embed_start_time = time.time()
                query_embedding = embeddings.embed_query(query)
                embed_duration = time.time() - embed_start_time
                
                query_norm = sum(x*x for x in query_embedding)**0.5
                logger.info(f"📊 查询嵌入完成 - 耗时: {embed_duration:.3f}秒 | 维度: {len(query_embedding)} | 范数: {query_norm:.4f}")
                
                # 测试文档嵌入
                test_text = "这是一个测试文档"
                doc_embed_start_time = time.time()
                doc_embedding = embeddings.embed_documents([test_text])
                doc_embed_duration = time.time() - doc_embed_start_time
                
                doc_norm = sum(x*x for x in doc_embedding[0])**0.5
                logger.info(f"📊 文档嵌入完成 - 耗时: {doc_embed_duration:.3f}秒 | 维度: {len(doc_embedding[0])} | 范数: {doc_norm:.4f}")
                
                total_debug_duration = time.time() - debug_start_time
                logger.info(f"✅ 嵌入模型测试完成 - 总耗时: {total_debug_duration:.3f}秒")
                
                return {
                    "embedding_test": {
                        "query_vector_norm": query_norm,
                        "doc_vector_norm": doc_norm,
                        "vector_dimension": len(query_embedding),
                        "query_embed_time": embed_duration,
                        "doc_embed_time": doc_embed_duration
                    }
                }
            except Exception as e:
                logger.error(f"❌ 嵌入模型测试失败: {str(e)}", exc_info=True)
                return {"embedding_test": {"error": str(e)}}
        
        # 使用自适应检索
        logger.info("🔍 开始调试检索过程...")
        relevant_docs, max_similarity, retrieval_info = adaptive_retrieval(
            vector_store, query, max_docs, 1, relevance_threshold
        )
        
        # 获取原始检索结果用于调试
        initial_k = min(max_docs * 2, 10)
        logger.debug(f"📊 获取原始检索结果 - K值: {initial_k}")
        
        raw_search_start_time = time.time()
        raw_docs = vector_store.similarity_search_with_score(query, k=initial_k)
        raw_search_duration = time.time() - raw_search_start_time
        
        logger.info(f"⚡ 原始检索完成 - 耗时: {raw_search_duration:.3f}秒 | 结果数: {len(raw_docs)}")
        
        # 构建调试信息
        debug_info = {
            "query": query,
            "retrieval_info": {
                "retrieved": retrieval_info.get("retrieved", 0),
                "relevant": retrieval_info.get("relevant", 0),
                "threshold": float(retrieval_info.get("threshold", 0.0)),
                "max_similarity": float(retrieval_info.get("max_similarity", 0.0)),
                "original_threshold": float(retrieval_info.get("original_threshold", 0.0)),
                "distance_stats": {
                    "min": float(retrieval_info.get("distance_stats", {}).get("min", 0.0)),
                    "max": float(retrieval_info.get("distance_stats", {}).get("max", 0.0)),
                    "avg": float(retrieval_info.get("distance_stats", {}).get("avg", 0.0))
                } if "distance_stats" in retrieval_info else None,
                "is_normalized": retrieval_info.get("is_normalized", False),
                "similarity_method": retrieval_info.get("similarity_method", "unknown"),
                "search_duration": retrieval_info.get("search_duration", 0.0),
                "similarity_duration": retrieval_info.get("similarity_duration", 0.0)
            },
            "raw_docs": []
        }
        
        logger.debug("📋 处理原始检索结果...")
        for i, (doc, score) in enumerate(raw_docs):
            try:
                # 确保score是Python原生类型
                score_float = float(score)
                similarity_score = calculate_similarity_score_optimized(score_float, method="sigmoid")
                
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                logger.debug(f"  文档 {i+1}: 距离 {score_float:.4f} -> 相似度 {similarity_score:.4f}")
                
                debug_info["raw_docs"].append({
                    "index": i,
                    "distance_score": score_float,
                    "similarity_score": float(similarity_score),
                    "content_preview": content_preview,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                })
            except Exception as e:
                logger.warning(f"⚠️  处理文档{i}时出错: {str(e)}")
                debug_info["raw_docs"].append({
                    "index": i,
                    "distance_score": float(score) if hasattr(score, '__float__') else 0.0,
                    "similarity_score": 0.0,
                    "content_preview": "处理出错",
                    "metadata": doc.metadata,
                    "error": str(e)
                })
        
        total_debug_duration = time.time() - debug_start_time
        logger.info(f"✅ RAG调试完成 - 总耗时: {total_debug_duration:.3f}秒 | 检索到: {len(raw_docs)}个 | 相关: {retrieval_info.get('relevant', 0)}个")
        
        return debug_info
        
    except Exception as e:
        total_debug_duration = time.time() - debug_start_time
        logger.error(f"❌ RAG调试失败 - 总耗时: {total_debug_duration:.3f}秒 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG调试失败: {str(e)}")

@app.post("/api/chat/rag", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """基于知识库的RAG对话接口"""
    session_start_time = time.time()
    logger.info(f"🚀 RAG对话请求开始 - 会话ID: {request.session_id} | 模型: {current_model} | 使用RAG: {request.use_rag}")
    logger.info(f"💬 用户问题: '{request.message[:100]}...' (长度: {len(request.message)})")
    
    try:
        # 检查模型是否初始化
        if llm is None:
            logger.error("❌ 模型未初始化")
            raise HTTPException(status_code=503, detail="模型未初始化，请检查Ollama服务")
        
        # 如果用户选择不使用RAG，则直接使用普通对话
        if not request.use_rag:
            logger.info("📝 用户选择不使用RAG，转为普通对话模式")
            # 获取/创建memory
            if not request.session_id:
                raise HTTPException(status_code=400, detail="缺少session_id")
            memory = session_memories.get(request.session_id)
            if memory is None:
                memory = ConversationBufferMemory(return_messages=True)
                session_memories[request.session_id] = memory
                logger.debug(f"🆕 创建新的会话记忆: {request.session_id}")
            
            # 构建干净的历史记录，避免RAG相关提示词残留
            clean_history = clean_conversation_history(memory)
            
            # 构建纯净的提示词
            clean_prompt = f"{request.system_prompt}\n\n{clean_history}用户：{request.message}\n助手："
            logger.debug(f"📝 普通对话提示词长度: {len(clean_prompt)}字符")
            
            # 打印最终提示词
            logger.info("🔍 最终提交给大模型的提示词:")
            logger.info("="*80)
            logger.info(clean_prompt)
            logger.info("="*80)
            
            # 运行LLM查询
            llm_start_time = time.time()
            result = llm.invoke(clean_prompt)
            llm_duration = time.time() - llm_start_time
            logger.info(f"🤖 LLM生成完成 - 耗时: {llm_duration:.3f}秒 | 响应长度: {len(result)}")
            
            # 更新memory
            memory.chat_memory.add_user_message(request.message)
            memory.chat_memory.add_ai_message(result)
            
            total_duration = time.time() - session_start_time
            logger.info(f"✅ 普通对话完成 - 总耗时: {total_duration:.3f}秒")
            
            return ChatResponse(response=result, model=current_model)
        
        if embeddings is None:
            logger.error("❌ 嵌入模型未初始化")
            raise HTTPException(status_code=503, detail="嵌入模型未初始化，无法进行RAG对话")
        
        if vector_store is None:
            logger.error("❌ 知识库未加载")
            raise HTTPException(status_code=503, detail="知识库未加载，请检查knowledge_base目录")
        
        # 获取/创建memory
        if not request.session_id:
            raise HTTPException(status_code=400, detail="缺少session_id")
        
        memory = session_memories.get(request.session_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            session_memories[request.session_id] = memory
            logger.debug(f"🆕 创建新的会话记忆: {request.session_id}")
        
        # 第一步：检索相关文档（使用优化后的检索函数）
        logger.info(f"🔍 开始RAG文档检索 - 参数: max_docs={request.max_docs}, threshold={request.relevance_threshold}")
        with performance_monitor("RAG文档检索"):
            relevant_docs, max_similarity, retrieval_info = optimized_retrieval(
                vector_store, 
                request.message, 
                max_docs=request.max_docs, 
                relevance_threshold=request.relevance_threshold
            )
        
        # 记录检索结果详细信息
        logger.info(f"📊 检索结果统计: {retrieval_info}")
        
        # 第二步：根据是否找到相关文档构建不同的提示词
        if relevant_docs:
            # 找到相关文档：基于知识库内容回答
            context_length = sum(len(doc.page_content) for doc in relevant_docs)
            logger.info(f"📚 构建知识库上下文 - 文档数: {len(relevant_docs)} | 总长度: {context_length}字符")
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"""基于以下知识库内容回答问题：

知识库内容：
{context}

用户问题：{request.message}

请基于知识库内容回答用户问题。要求：
1. 优先使用知识库中提供的信息
2. 如果知识库中没有相关信息，请明确说明"知识库中没有相关信息"
3. 回答要简洁准确，不要重复知识库内容
4. 不要添加知识库中没有的信息

回答："""
            logger.debug(f"📝 RAG提示词长度: {len(prompt)}字符")
        else:
            # 没有找到相关文档：直接让模型回答
            logger.warning("⚠️  未找到相关文档，转为通用知识回答")
            prompt = f"""用户问题：{request.message}

请用简洁明了的语言回答用户的问题。如果涉及专业知识，请基于你的训练数据回答。

注意：知识库中没有找到与用户问题直接相关的内容，请基于你的通用知识来回答。"""
        
        # 打印最终提示词
        logger.info("🔍 最终提交给大模型的RAG提示词:")
        logger.info("="*80)
        logger.info(prompt)
        logger.info("="*80)
        
        # 运行LLM查询
        logger.info("🤖 开始LLM生成回答...")
        with performance_monitor("LLM生成"):
            llm_start_time = time.time()
            result = llm.invoke(prompt)
            llm_duration = time.time() - llm_start_time
            logger.info(f"🤖 LLM生成完成 - 耗时: {llm_duration:.3f}秒 | 响应长度: {len(result) if result else 0}")
        
        # 构建响应
        response_text = result if result else "抱歉，未能生成有效响应。"
        
        # 更新memory
        memory.chat_memory.add_user_message(request.message)
        memory.chat_memory.add_ai_message(response_text)
        
        total_duration = time.time() - session_start_time
        logger.info(f"✅ RAG对话完成 - 总耗时: {total_duration:.3f}秒 | 检索耗时: {retrieval_info.get('search_duration', 0):.3f}秒 | LLM耗时: {llm_duration:.3f}秒")
        
        return ChatResponse(response=response_text, model=current_model)
        
    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - session_start_time
        logger.error(f"❌ RAG对话失败 - 总耗时: {total_duration:.3f}秒 | 错误: {str(e)}", exc_info=True)
        return ChatResponse(
            response="",
            model=current_model,
            success=False,
            error=f"RAG对话失败: {str(e)}"
        )

# RAG流式聊天接口
@app.post("/api/chat/rag/stream")
async def chat_with_rag_stream(request: ChatRequest):
    """基于知识库的RAG流式对话接口"""
    session_start_time = time.time()
    logger.info(f"🚀 RAG流式对话请求开始 - 会话ID: {request.session_id} | 模型: {current_model} | 使用RAG: {request.use_rag}")
    logger.info(f"💬 用户问题: '{request.message[:100]}...' (长度: {len(request.message)})")
    
    try:
        # 检查模型是否初始化
        if llm is None:
            logger.error("❌ 模型未初始化")
            raise HTTPException(status_code=503, detail="模型未初始化，请检查Ollama服务")
        
        # 如果用户选择不使用RAG，则直接使用普通对话
        if not request.use_rag:
            logger.info("📝 用户选择不使用RAG，转为普通流式对话模式")
            # 获取/创建memory
            if not request.session_id:
                raise HTTPException(status_code=400, detail="缺少session_id")
            memory = session_memories.get(request.session_id)
            if memory is None:
                memory = ConversationBufferMemory(return_messages=True)
                session_memories[request.session_id] = memory
                logger.debug(f"🆕 创建新的会话记忆: {request.session_id}")
            
            # 构建干净的历史记录，避免RAG相关提示词残留
            clean_history = clean_conversation_history(memory)
            
            # 构建纯净的提示词
            clean_prompt = f"{request.system_prompt}\n\n{clean_history}用户：{request.message}\n助手："
            logger.debug(f"📝 普通流式提示词长度: {len(clean_prompt)}字符")
            
            # 打印最终提示词
            logger.info("🔍 最终提交给大模型的流式提示词:")
            logger.info("="*80)
            logger.info(clean_prompt)
            logger.info("="*80)
            
            async def event_generator():
                try:
                    chunk_count = 0
                    empty_chunks = 0
                    full_response = ""
                    stream_start_time = time.time()
                    logger.info("🌊 开始普通流式生成...")
                    
                    for chunk in llm.stream(clean_prompt):
                        if chunk:
                            chunk_count += 1
                            empty_chunks = 0
                            full_response += chunk
                            yield ServerSentEvent(data=chunk, event="message")
                            await asyncio.sleep(0.01)
                        else:
                            empty_chunks += 1
                            if empty_chunks > 10:
                                logger.warning("⚠️  连续空块过多，终止流式响应")
                                yield ServerSentEvent(data="", event="error", id="empty_chunks")
                                break
                    
                    stream_duration = time.time() - stream_start_time
                    total_duration = time.time() - session_start_time
                    
                    if chunk_count == 0:
                        logger.error("❌ 未收到有效流式响应块")
                        yield ServerSentEvent(data="未收到有效响应", event="error", id="no_content")
                    else:
                        logger.info(f"✅ 普通流式对话完成 - 总耗时: {total_duration:.3f}秒 | 流式耗时: {stream_duration:.3f}秒 | 响应长度: {len(full_response)} | 块数: {chunk_count}")
                        yield ServerSentEvent(data="", event="end")
                        
                    # 更新memory
                    memory.chat_memory.add_user_message(request.message)
                    memory.chat_memory.add_ai_message(full_response)
                        
                except Exception as e:
                    logger.error(f"❌ 普通流式对话出错: {str(e)}", exc_info=True)
                    yield ServerSentEvent(data=f"处理出错: {str(e)}", event="error")
            
            return EventSourceResponse(event_generator())
        
        if embeddings is None:
            logger.error("❌ 嵌入模型未初始化")
            raise HTTPException(status_code=503, detail="嵌入模型未初始化，无法进行RAG对话")
        
        if vector_store is None:
            logger.error("❌ 知识库未加载")
            raise HTTPException(status_code=503, detail="知识库未加载，请检查knowledge_base目录")
        
        # 获取/创建memory
        if not request.session_id:
            raise HTTPException(status_code=400, detail="缺少session_id")
        
        memory = session_memories.get(request.session_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            session_memories[request.session_id] = memory
            logger.debug(f"🆕 创建新的会话记忆: {request.session_id}")
        
        # 第一步：检索相关文档（使用优化后的检索函数）
        logger.info(f"🔍 开始RAG文档检索 - 参数: max_docs={request.max_docs}, threshold={request.relevance_threshold}")
        relevant_docs, max_similarity, retrieval_info = optimized_retrieval(
            vector_store, 
            request.message, 
            max_docs=request.max_docs, 
            relevance_threshold=request.relevance_threshold
        )
        
        # 记录检索结果详细信息
        logger.info(f"📊 检索结果统计: {retrieval_info}")
        
        # 第二步：根据是否找到相关文档构建不同的提示词
        if relevant_docs:
            # 找到相关文档：基于知识库内容回答
            context_length = sum(len(doc.page_content) for doc in relevant_docs)
            logger.info(f"📚 构建知识库上下文 - 文档数: {len(relevant_docs)} | 总长度: {context_length}字符")
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"""基于以下知识库内容回答问题：

知识库内容：
{context}

用户问题：{request.message}

请基于知识库内容回答用户问题。要求：
1. 优先使用知识库中提供的信息
2. 如果知识库中没有相关信息，请明确说明"知识库中没有相关信息"
3. 回答要简洁准确，不要重复知识库内容
4. 不要添加知识库中没有的信息

回答："""
            logger.debug(f"📝 RAG提示词长度: {len(prompt)}字符")
        else:
            # 没有找到相关文档：直接让模型回答
            logger.warning("⚠️  未找到相关文档，转为通用知识回答")
            prompt = f"""用户问题：{request.message}

请用简洁明了的语言回答用户的问题。如果涉及专业知识，请基于你的训练数据回答。

注意：知识库中没有找到与用户问题直接相关的内容，请基于你的通用知识来回答。"""
        
        # 打印最终提示词
        logger.info("🔍 最终提交给大模型的RAG流式提示词:")
        logger.info("="*80)
        logger.info(prompt)
        logger.info("="*80)
        
        async def event_generator():
            try:
                chunk_count = 0
                empty_chunks = 0
                full_response = ""
                stream_start_time = time.time()
                logger.info("🌊 开始RAG流式生成...")
                
                for chunk in llm.stream(prompt):
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
                
                stream_duration = time.time() - stream_start_time
                total_duration = time.time() - session_start_time
                
                if chunk_count == 0:
                    logger.error("❌ 未收到有效流式响应块")
                    yield ServerSentEvent(data="未收到有效响应", event="error", id="no_content")
                else:
                    logger.info(f"✅ RAG流式对话完成 - 总耗时: {total_duration:.3f}秒 | 检索耗时: {retrieval_info.get('search_duration', 0):.3f}秒 | 流式耗时: {stream_duration:.3f}秒 | 响应长度: {len(full_response)} | 块数: {chunk_count}")
                    yield ServerSentEvent(data="", event="end")
                    
                # 更新memory
                memory.chat_memory.add_user_message(request.message)
                memory.chat_memory.add_ai_message(full_response)
                    
            except Exception as e:
                logger.error(f"❌ RAG流式处理出错: {str(e)}", exc_info=True)
                yield ServerSentEvent(data=f"处理出错: {str(e)}", event="error")
        
        return EventSourceResponse(event_generator())
        
    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - session_start_time
        logger.error(f"❌ RAG流式对话失败 - 总耗时: {total_duration:.3f}秒 | 错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG流式对话失败: {str(e)}")

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
        
        # 构建干净的历史记录，避免RAG相关提示词残留
        clean_history = clean_conversation_history(memory)
        
        # 构建纯净的提示词
        clean_prompt = f"{request.system_prompt}\n\n{clean_history}用户：{request.message}\n助手："
        logger.debug(f"📝 普通对话提示词长度: {len(clean_prompt)}字符")
        
        # 打印最终提示词
        logger.info("🔍 最终提交给大模型的普通对话提示词:")
        logger.info("="*80)
        logger.info(clean_prompt)
        logger.info("="*80)
        
        # 运行LLM查询
        result = llm.invoke(clean_prompt)
        
        # 验证结果
        if not result or not result.strip():
            logger.warning("模型返回空响应")
            return ChatResponse(
                response="抱歉，未能获取到有效响应。",
                model=current_model,
                success=False,
                error="模型返回空响应"
            )
        
        logger.info(f"模型响应: {result[:100]}...")  # 只打印前100个字符
        # 更新memory
        memory.chat_memory.add_user_message(request.message)
        memory.chat_memory.add_ai_message(result)
        return ChatResponse(response=result, model=current_model)
    
    except HTTPException:
        # 重新抛出FastAPI HTTP异常
        raise
    except Exception as e:
        # 捕获所有其他异常并记录
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        # 返回结构化错误响应
        return ChatResponse(
            response="",
            model=current_model,
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
        
        # 构建干净的历史记录，避免RAG相关提示词残留
        history = clean_conversation_history(memory)
        
        # 创建提示模板
        prompt = f"{request.system_prompt}\n\n"
        # ConversationChain自动注入历史
        full_prompt = prompt + "{history}\n用户问：{input}\n助手答："
        prompt_filled = full_prompt.replace("{history}", history).replace("{input}", request.message)
        
        # 打印最终提示词
        logger.info("🔍 最终提交给大模型的普通流式提示词:")
        logger.info("="*80)
        logger.info(prompt_filled)
        logger.info("="*80)
        
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
                logger.error(f"处理流式请求时出错: {str(e)}", exc_info=True)
                yield ServerSentEvent(data=f"处理出错: {str(e)}", event="error")
        
        return EventSourceResponse(event_generator())
    
    except HTTPException:
        # 重新抛出FastAPI HTTP异常
        raise
    except Exception as e:
        logger.error(f"处理流式请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理流式请求时出错: {str(e)}")

# 性能分析API
@app.post("/api/performance/analyze")
async def analyze_rag_performance(
    request: dict
):
    """分析RAG性能瓶颈"""
    try:
        # 从请求体中提取参数
        query = request.get("query", "什么是人工智能？")
        max_docs = request.get("max_docs", 5)
        relevance_threshold = request.get("relevance_threshold", 0.5)
        
        if vector_store is None:
            raise HTTPException(status_code=503, detail="知识库未加载")
        
        if embeddings is None:
            raise HTTPException(status_code=503, detail="嵌入模型未初始化")
        
        performance_data = {}
        
        # 测试嵌入模型性能
        with performance_monitor("嵌入模型测试"):
            try:
                query_embedding = embeddings.embed_query(query)
                performance_data["embedding_time"] = "正常"
                performance_data["embedding_dimension"] = len(query_embedding)
            except Exception as e:
                performance_data["embedding_time"] = f"失败: {str(e)}"
        
        # 测试向量检索性能
        with performance_monitor("向量检索测试"):
            try:
                docs = vector_store.similarity_search_with_score(query, k=max_docs)
                performance_data["retrieval_time"] = "正常"
                performance_data["retrieved_docs"] = len(docs)
            except Exception as e:
                performance_data["retrieval_time"] = f"失败: {str(e)}"
        
        # 测试相似度计算性能
        if "retrieved_docs" in performance_data and performance_data["retrieved_docs"] > 0:
            with performance_monitor("相似度计算测试"):
                try:
                    distances = []
                    for result in docs:
                        if isinstance(result, tuple) and len(result) == 2:
                            distances.append(float(result[1]))
                    
                    if distances:
                        # 测试相似度计算
                        for distance in distances[:3]:  # 只测试前3个
                            calculate_similarity_score_optimized(distance)
                        performance_data["similarity_calculation"] = "正常"
                    else:
                        performance_data["similarity_calculation"] = "无有效距离分数"
                except Exception as e:
                    performance_data["similarity_calculation"] = f"失败: {str(e)}"
        
        # 测试LLM性能
        with performance_monitor("LLM测试"):
            try:
                test_prompt = f"请简单回答：{query}"
                result = llm.invoke(test_prompt)
                performance_data["llm_time"] = "正常"
                performance_data["llm_response_length"] = len(result) if result else 0
            except Exception as e:
                performance_data["llm_time"] = f"失败: {str(e)}"
        
        # 完整RAG流程测试
        with performance_monitor("完整RAG流程"):
            try:
                relevant_docs, max_similarity, retrieval_info = optimized_retrieval(
                    vector_store, query, max_docs, relevance_threshold
                )
                performance_data["full_rag_time"] = "正常"
                performance_data["relevant_docs_count"] = len(relevant_docs)
                performance_data["max_similarity"] = max_similarity
            except Exception as e:
                performance_data["full_rag_time"] = f"失败: {str(e)}"
        
        return {
            "status": "success",
            "performance_data": performance_data,
            "recommendations": generate_performance_recommendations(performance_data)
        }
        
    except Exception as e:
        logger.error(f"性能分析失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"性能分析失败: {str(e)}")

def generate_performance_recommendations(performance_data: dict) -> list:
    """根据性能数据生成优化建议"""
    recommendations = []
    
    # 检查嵌入模型性能
    if "embedding_time" in performance_data and "失败" in performance_data["embedding_time"]:
        recommendations.append("嵌入模型响应慢，建议检查Ollama服务状态")
    
    # 检查向量检索性能
    if "retrieval_time" in performance_data and "失败" in performance_data["retrieval_time"]:
        recommendations.append("向量检索失败，建议检查FAISS索引状态")
    
    # 检查LLM性能
    if "llm_time" in performance_data and "失败" in performance_data["llm_time"]:
        recommendations.append("LLM响应慢，建议检查Ollama服务状态")
    
    # 检查完整流程性能
    if "full_rag_time" in performance_data and "失败" in performance_data["full_rag_time"]:
        recommendations.append("完整RAG流程失败，建议检查各组件状态")
    
    # 如果没有明显问题，提供一般性建议
    if not recommendations:
        recommendations.extend([
            "性能表现正常",
            "如需进一步优化，可考虑：",
            "- 减少检索文档数量",
            "- 调整相关性阈值",
            "- 使用更快的嵌入模型",
            "- 优化向量索引"
        ])
    
    return recommendations

@app.get("/api/health", description="检查API和模型健康状态")
async def health_check():
    """检查API和模型是否正常运行"""
    try:
        # 检查模型是否初始化
        if llm is None:
            return {
                "status": "unhealthy",
                "model": current_model,
                "available_models": available_models,
                "message": "模型未初始化，请检查Ollama服务",
                "test_response": None,
                "working_directory": str(WORKING_DIR.absolute()),
                "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR.absolute()),
                "vector_stores_dir": str(VECTOR_STORES_DIR.absolute())
            }
        
        # 简单测试模型是否可用
        test_response = llm.invoke("你好")
        return {
            "status": "healthy",
            "model": current_model,
            "available_models": available_models,
            "message": f"API和模型({current_model})均正常运行",
            "test_response": test_response[:50],  # 返回部分测试响应
            "working_directory": str(WORKING_DIR.absolute()),
            "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR.absolute()),
            "vector_stores_dir": str(VECTOR_STORES_DIR.absolute())
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "model": current_model,
            "available_models": available_models,
            "message": f"模型不可用: {str(e)}",
            "test_response": None,
            "working_directory": str(WORKING_DIR.absolute()),
            "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR.absolute()),
            "vector_stores_dir": str(VECTOR_STORES_DIR.absolute())
        }

# 添加历史记录清理函数
def clean_conversation_history(memory: ConversationBufferMemory) -> str:
    """
    清理对话历史，移除RAG相关的提示词和内容
    """
    clean_history = ""
    rag_keywords = [
        "基于以下知识库内容", "根据文档内容", "知识库中", "基于知识库", "文档中提到",
        "请基于知识库内容回答", "知识库内容：", "根据提供的文档", "文档显示",
        "基于上述文档", "文档资料显示", "根据相关文档", "文档信息表明"
    ]
    
    for m in memory.chat_memory.messages:
        if m.type == "human":
            clean_history += f"用户：{m.content}\n"
        elif m.type == "ai":
            ai_content = m.content
            # 检查是否包含RAG相关关键词
            is_rag_response = any(keyword in ai_content for keyword in rag_keywords)
            
            # 如果不是RAG响应，或者是经过清理的响应，则保留
            if not is_rag_response:
                clean_history += f"助手：{ai_content}\n"
            else:
                logger.debug(f"过滤RAG响应: {ai_content[:50]}...")
    
    return clean_history

if __name__ == "__main__":
    import uvicorn
    logger.info("启动API服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

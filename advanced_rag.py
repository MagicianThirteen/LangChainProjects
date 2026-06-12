from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import init_chat_model
import logging

load_dotenv()

#通过打开multiqueryretriever的日志来看，到底扩展了哪些问题
logging.basicConfig(level=logging.INFO,format="%(name)s---%(message)s")
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

# Sample knowledge base for demos
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

INFO_BURIED = [
    Document(
        page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK

Founded in 2018 by three Stanford graduates, ACME AI Solutions began as a
small consulting firm helping enterprises adopt machine learning. Our first
office was a converted garage in Palo Alto, and we had just two laptops and
a dream. The early days were challenging - we survived on instant ramen and
the occasional pizza from the client meetings.

In 2019, we secured our first major contract with a Fortune 500 retailer,
helping them build a recommendation engine. This led to rapid growth and we
moved to a proper office space in San Francisco. By 2020, we had grown to
50 employees and opened offices in Austin and Seattle.

Our current technology stack has evolved significantly over the years. For
backend services, we use Python and FastAPI. Our data pipeline runs on
Apache Spark and Airflow. For frontend, we've standardized on React and
TypeScript.

LangChain is a framework for building LLM applications. It provides tools
for prompts, chains, agents, and memory. LangChain supports multiple LLM
providers including OpenAI, Anthropic, and local models like Llama.

The company culture at ACME emphasizes work-life balance. We offer unlimited
PTO, which most employees use for an average of 25 days per year. Our
engineering teams follow agile methodology with two-week sprints.

Our revenue has grown consistently, from $2M in 2019 to $45M in 2023. We
project $70M for 2024, driven by our new enterprise AI platform. The company
went through Series B funding in 2022, raising $80M at a $500M valuation.

Employee benefits include comprehensive health insurance through Aetna, a
401(k) with 4% matching, and a generous equity package.""",
        metadata={"source": "acme_company_overview.pdf"},
    ),
    Document(
        page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4

Chapter 1: System Architecture Overview

The ACME AI Platform is built on a microservices architecture deployed on
AWS EKS (Elastic Kubernetes Service). Each microservice is containerized
using Docker and orchestrated by Kubernetes. We use Istio as our service
mesh for traffic management and observability.

Our database layer consists of PostgreSQL for transactional data, Redis
for caching, and Pinecone for vector storage. All databases are deployed
in high-availability configurations with automatic failover.

Chapter 2: Authentication and Authorization

User authentication is handled through Auth0, supporting both SSO via SAML
2.0 and OAuth 2.0 flows. We implement role-based access control (RBAC) with
four default roles: Admin, Developer, Analyst, and Viewer.

Chapter 3: AI Framework Integration

LangGraph is a library for building stateful, multi-actor applications with
LLMs. Key features include state management, cycles and loops, human-in-the-
loop workflows, and persistence. LangGraph extends LangChain for complex
agent architectures.

Chapter 4: Monitoring and Logging

We use DataDog for application performance monitoring (APM) and log
aggregation. All services emit structured JSON logs that are collected and
indexed for searching. Alert thresholds are configured for latency (p99 >
500ms), error rates (> 1%), and resource utilization (CPU > 80%).

Chapter 5: Disaster Recovery

Our disaster recovery plan includes daily database backups stored in S3
with cross-region replication. RTO is 4 hours, and RPO is 1 hour.""",
        metadata={"source": "technical_docs_v2.4.pdf"},
    ),
]

embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
llm=init_chat_model(model="gpt-4o-mini",temperature=0.2)
def return_vectorstore(documents):
    vectorstore=Chroma.from_documents(
        documents=documents,
        embedding=embedding_model
    )
    return vectorstore



def multi_query_retriever():
    #返回一个向量数据库chroma
    vectorstore=return_vectorstore(TECH_DOCS)
    #设置好ritriever
    ritriever=vectorstore.as_retriever(
        search_kwargs={"k":2}
    )
    #调用multiqueryretriever
    multiritriever=MultiQueryRetriever.from_llm(
        retriever=ritriever,
        llm=llm
    )
    query = "What tools can I use to build AI applications?"
    #打印文档的topic和content
    docs=multiritriever.invoke(query)
    for i,doc in enumerate(docs):
        print(f"[{i+1}][{doc.metadata.get('topic','N/A')}]{doc.page_content[:50]}")
    
    '''
    httpx---HTTP Request: POST https://poloai.top/v1/embeddings "HTTP/1.1 200 OK"
httpx---HTTP Request: POST https://poloai.top/v1/chat/completions "HTTP/1.1 200 OK"
langchain_classic.retrievers.multi_query---Generated queries: ['What are some recommended tools for developing AI applications?  ', 'Which software or platforms are best suited for creating AI applications?  ', 'Can you suggest tools that are effective for building applications in artificial intelligence?']
httpx---HTTP Request: POST https://poloai.top/v1/embeddings "HTTP/1.1 200 OK"
httpx---HTTP Request: POST https://poloai.top/v1/embeddings "HTTP/1.1 200 OK"
httpx---HTTP Request: POST https://poloai.top/v1/embeddings "HTTP/1.1 200 OK"
[1][ai]Machine learning is a subset of AI that enables sy
[2][ai]LangChain is a framework for building LLM applicat
[3][database]Vector databases like Pinecone, Chroma, and Qdrant
    '''


def contextual_compression_retriever():
    #返回数据库
    vectorstore=return_vectorstore(INFO_BURIED)
    #定义普通的retriever，相似度检索，3
    base_retriever=vectorstore.as_retriever(
        search_kwargs={"k":3}
    )
    #获取一个llm给的压缩器
    compressor=LLMChainExtractor.from_llm(llm)
    #生成一个带文档压缩器的retriever
    compress_retriever=ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    query = "What frameworks exist for building LLM applications?"
    #invoke查看结果
    docs=compress_retriever.invoke(query)
    for doc in docs:
        print(f"{len(doc.page_content)}\n")
        print(f"{doc.page_content[:150]}")
    
    '''
    httpx---HTTP Request: POST https://poloai.top/v1/embeddings "HTTP/1.1 200 OK"
httpx---HTTP Request: POST https://poloai.top/v1/embeddings "HTTP/1.1 200 OK"
httpx---HTTP Request: POST https://poloai.top/v1/chat/completions "HTTP/1.1 200 OK"
httpx---HTTP Request: POST https://poloai.top/v1/chat/completions "HTTP/1.1 200 OK"
214

LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM pro
283

Chapter 3: AI Framework Integration

LangGraph is a library for building stateful, multi-actor applications with
LLMs. Key features include state mana
    
    '''


if __name__ == "__main__":
   # multi_query_retriever()
   contextual_compression_retriever()




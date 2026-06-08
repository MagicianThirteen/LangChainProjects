from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv

load_dotenv()
embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory="./chroma_db"

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def chroma_basics():
        #创建数据库
        vectorstore=Chroma.from_documents(
            embedding=embedding_model,
            documents=SAMPLE_DOCS,
            persist_directory="./chroma_db"
        )
        print(f"当前录入了{vectorstore._collection.count()}个数据")
        #设置问题
        query = "What is LangChain?"
        #最相似搜索
        result=vectorstore.similarity_search(query,k=2)
        for i,doc in enumerate(result):
            print(f"第{i+1}个文档的内容是{doc.page_content}\n 它的元数据是:{doc.metadata}")

def chroma_search_with_scores():
     query = "Explain vector stores."
     #定义数据库
     vectorstore=Chroma.from_documents(
          embedding=embedding_model,
          persist_directory="./chroma_db",
          documents=SAMPLE_DOCS
     )
     #数据库调用查找函数
     result=vectorstore.similarity_search_with_score(query,k=2)
     #打印
     for i,(doc,score) in enumerate(result):
          #print(f"第{i+1}个文件的内容是:{doc.page_content}\n分数是{score:.4f}\n来源是{doc.metadata['source']}")
          print(
        f"第{i+1}个文件的内容是:{doc.page_content}\n"
        f"分数是{score:.4f}\n"
        f"来源是{doc.metadata['source']}"
    )

def persist_chroma():
    vectorstore=Chroma.from_documents(
         embedding=embedding_model,
         persist_directory=persist_directory,
         documents=SAMPLE_DOCS
    )

    print(f"当前有{vectorstore._collection.count()}个数据")

    del vectorstore

    reloaded_store=Chroma(
         embedding_function=embedding_model,
         persist_directory=persist_directory
    ) 
    reloaded_count=reloaded_store._collection.count()
    print(f"重新加载后当前有{reloaded_count}个数据")

    result=reloaded_store.similarity_search(query="Langchain",k=2,
                                            filter={"topic": "overview"})
    for c in result:
         print(f"{c.page_content[:50]}\n")
    
    '''
    当前有8个数据
    重新加载后当前有8个数据
    LangChain is a framework for developing applicatio

    LangGraph is a library for building stateful, mult
    '''





if __name__ == "__main__":
   # chroma_basics()
   #chroma_search_with_scores()
   persist_chroma()
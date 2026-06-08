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

def as_retriever():
    vectorstore=Chroma.from_documents(
         documents=SAMPLE_DOCS,
         persist_directory="./chroma.db",
         embedding=embedding_model
    )
    s_retriever=vectorstore.as_retriever(
         search_type="similarity",
         search_kwargs={"k":3}
    )
    s_result=s_retriever.invoke("LangChain")
    for doc in s_result:
         print(f"相似检索的内容：{doc.page_content}")
    '''
    相似检索的内容：LangChain is a framework for developing applications powered by language models.
    相似检索的内容：LangGraph is a library for building stateful, multi-actor applications with LLMs.
    相似检索的内容：Chroma is an open-source embedding database for AI applications.
    '''

    m_retriever=vectorstore.as_retriever(
         search_type="mmr",
         search_kwargs={"k":3,"fecth_k":5}
    )
    m_result=m_retriever.invoke("AI applications")
    for doc in m_result:
        print(f"mmr相关检索的内容：{doc.page_content}")
    
    '''
    mmr相关检索的内容：Chroma is an open-source embedding database for AI applications.
    mmr相关检索的内容：FAISS is a library for efficient similarity search developed by Facebook.
    mmr相关检索的内容：LangGraph is a library for building stateful, multi-actor applications with LLMs.
    
    '''

def vector_store_use():
    sample_texts = [
        "Python is a versatile programming language used in web development, "
        "data science, machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read.",
        "JavaScript is the language of the web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient.",
        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time.",
    ] 

    queries = [
        "What's good for web development?",
        "Which language is safest?",
    ]

    #把list[str]=>list[document]
    documents=[Document(page_content=c)for c in sample_texts]
    #制作分割器，其中list[document],chunk_size,chunk_overlap
    def getchunks(documents,chunk_size,chunk_overlap):
        spliter=RecursiveCharacterTextSplitter(
             chunk_size=chunk_size,
             chunk_overlap=chunk_overlap
        )
        chunks=spliter.split_documents(documents)
        return chunks
    #作为参数
    #把分割好的chunk（list[document]）放入建好的数据库
    chunks=getchunks(documents,200,50)
    vectorstore=Chroma.from_documents(
         documents=chunks,
         persist_directory="./tdb",
         embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    #通过数据库检索(retriever)
    retriever=vectorstore.as_retriever(
         search_type="similarity",
         search_kwargs={"k":2}
    )
    for q in queries:
        result=retriever.invoke(q)
        for r in result:
            print(f"{r.page_content}")
'''
ding web applications efficient.
Python is a versatile programming language used in web development, data science, machine learning, and automation. It has a simple syntax that makes it easy to learn and read.
Rust is a systems programming language focused on safety and performance. It prevents common bugs like null pointer dereferences and data races at compile time.
JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React and Vue make building web applications efficient.


'''
         




if __name__ == "__main__":
   # chroma_basics()
   #chroma_search_with_scores()
   #persist_chroma()
   #as_retriever()
   vector_store_use()
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.documents import Document


load_dotenv()
embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
llm=init_chat_model(model="gpt-4o-mini",temperature=0.2)
parser=StrOutputParser()


#基本文本信息
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""
#告诉大模型的格式
"""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know.
"""
#要提问的问题
def basic_rag():
    questions = [
            "What is LangChain?",
            "Who created LangChain?",
            "What is LangGraph used for?",
        ]

    #从文档检索信息，再把上下文和问题给大模型，然后返回结果
    head=[("#","h1"),("##","h2")]
    spliter=MarkdownHeaderTextSplitter(
        headers_to_split_on=head
    )
    chunks=spliter.split_text(KNOWLEDGE_BASE)
    vectorstore=Chroma.from_documents(
        embedding=embedding_model,
        persist_directory="./rag_db",
        documents=chunks
    )
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    prompt=ChatPromptTemplate.from_template(
        '''
        Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer:


    Make sure to answer in a concise manner, 
    and if you don't know the answer, just say "I don't know.
    '''
    )

    #为什么这里不需要先retriever.invoke?
    def format_doc(documents):
        return "\n\n".join([doc.page_content for doc in documents])

    chain={
        "context":retriever|format_doc,
        "question":RunnablePassthrough()
    }|prompt|llm|parser

    for q in questions:
        result=chain.invoke(q)
        print(f"llm返回：{result}")

'''
llm返回：LangChain is a framework for developing applications powered by language models, created by Harrison Chase in October 2022.
llm返回：Harrison Chase created LangChain.
llm返回：LangGraph is used for building stateful, multi-actor applications.
'''
#做个单独返回数据库的函数
def return_vectorstore(text,chunksize,chunkoverlap,embeddingmodel):
    document=[Document(
        page_content=text,
        metadata={"source":"Langchan.md"}#这里还可以做的更通用点
    )]
    spliter=RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
    )
    chunks=spliter.split_documents(document)
    vectorstore=Chroma.from_documents(
        documents=chunks,
        persist_directory="./ragdb",
        embedding=embeddingmodel
    )
    return vectorstore
    
#做个解析document成指定格式字符串的函数
def format_doc_with_source(documents):
    format_docs=[]
    for i,doc in enumerate(documents):
        source=doc.metadata.get("source","unknow")
        content=(f"{i+1}{source}:\n{doc.page_content}")
        format_docs.append(content)
    return "\n\n".join(format_docs)
#步骤打印
def log_step(x):
    print(x)
    print("打印喂给prompt的数据")
    return x

def rag_with_source():
    vectorstore=return_vectorstore(KNOWLEDGE_BASE
                                   ,500,50,embedding_model)
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":2}
    )
    prompt=ChatPromptTemplate.from_template(
        """
    Answer the question based on the context below. Include which sources you used.

    Context:
    {context}

    Question: {question}

    Answer (include sources):"""
    )

    chain=({
        "context":retriever|format_doc_with_source,
        "question":RunnablePassthrough()
    })|RunnableLambda(lambda x:log_step(x))|prompt|llm|parser

    print("RAG with Sources:\n")
    answer =chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")

'''
输出：
RAG with Sources:

{'context': '1Langchan.md:\n# LangChain Framework\n\nLangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.\n\n## Core Components\n\n1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.\n\n2. **Prompts**: Templates for structuring inputs to language models.\n\n3. **Chains**: Sequences of calls to models and other components.\n\n4. **Agents**: Systems that use LLMs to determine which actions to take.\n\n2Langchan.md:\n5. **Memory**: Components for persisting state between chain/agent calls.\n\n## LangGraph\n\nLangGraph is a library for building stateful, multi-actor applications. Key features:\n- State management\n- Cycles and loops\n- Human-in-the-loop\n- Persistence\n\n## Pricing\n\nLangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.\n\n## Getting Started', 'question': 'What are the core components of LangChain?'}
打印喂给prompt的数据
Q: What are the core components?

A: The core components of LangChain are:

1. **Models**: Supports various LLM providers including OpenAI, Anthropic, and local models.
2. **Prompts**: Templates for structuring inputs to language models.
3. **Chains**: Sequences of calls to models and other components.
4. **Agents**: Systems that use LLMs to determine which actions to take.
5. **Memory**: Components for persisting state between chain/agent calls.

(Source: 1Langchan.md)

'''

def structed_rag():
    #问题："What is LangGraph?"
    #返回数据库
    vectorstore=return_vectorstore(KNOWLEDGE_BASE,200,50,embedding_model)
    #返回检索器
    retiever=vectorstore.as_retriever(
        search_kwargs={"k":2}
    )
    #定义要输出的结构的类
    class RAGResponse(BaseModel):
        answer:str=Field(description="The answer to the question")
        confidence:str=Field(description="high, medium, or low")
        source_used:list[str]=Field(description="List of sources referenced")
        follow_up:str=Field(description="Suggested follow-up question")

    #llm要结构输出
    structed_llm=llm.with_structured_output(RAGResponse)
    #prompt的提示
    prompt=ChatPromptTemplate.from_template(
         """
Based on the context below, answer the question.

Context:
{context}

Question: {question}

Provide a structured response."""
    )
    #组合
    chain={"context":retiever|format_doc_with_source,
           "question":RunnablePassthrough()}|prompt|structed_llm
    result=chain.invoke("What is LangGraph?")
    #为什么这里不能这样写？result=chain.invoke({"question":"What is LangGraph?"})    
    #打印结果
    print(f"answer:{result.answer}")
    print(f"confidence:{result.confidence}")
    print(f"sources_used:{result.source_used}")
    print(f"follow_up:{result.follow_up}")

    '''
    answer:LangGraph is a library designed for creating stateful, multi-actor applications. It includes key features such as state management, the ability to handle cycles and loops, support for human-in-the-loop processes, and persistence of state across interactions.
    confidence:high
    sources_used:['1Langchan.md', '2Langchan.md']
    follow_up:What are some use cases for LangGraph?
    
    
    '''






if __name__ == "__main__":
    #rag_with_source()
    structed_rag()


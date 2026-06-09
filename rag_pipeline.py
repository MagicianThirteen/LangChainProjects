from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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

if __name__ == "__main__":
    basic_rag()


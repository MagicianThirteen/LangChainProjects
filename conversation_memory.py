from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from typing import Dict
from langchain_core.chat_history import(
    InMemoryChatMessageHistory,
    BaseChatMessageHistory
)
from langchain_core.messages import(
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()


llm=init_chat_model(model="gpt-4o-mini")
parser=StrOutputParser()
def basic_memory():
    prompt=ChatPromptTemplate.from_messages(
        [
            #特别注意，这里如果human和history这里调换了顺序，得出的结果并不一样
            #调换顺序，有的时候并不会按照你想回答的方式回答
            #为什么
            ("system","你是个很有用的助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human","{input}"),
            
        ]
    )
    chain=prompt|llm|parser

    #存在内存的历史数据库
    store:Dict[str,InMemoryChatMessageHistory]={}
    #获取历史记录的方法
    def get_session_id(session_id:str)->BaseChatMessageHistory:
        if session_id  not in store:
            store[session_id]=InMemoryChatMessageHistory()
        return store[session_id]
    
    #增强带有记忆的chain
    chain_with_history=RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_id,
        input_messages_key="input",
        history_messages_key="history"
    )

    messages = [
        "Hi! My name is Paulo.",
        "I'm learning about LangChain.",
        "What's my name and what am I learning?",
    ]
    for q in messages:
        response=chain_with_history.invoke(
            {"input":q},
            config={"configurable":{"session_id":"user_agent"}}
        )
        print(f"ai回答{response}")

    print(f"打印当前数据库里存的历史信息")
    for msg in store["user_agent"].messages:
        role="human" if isinstance(msg,HumanMessage) else "ai"
        print(f"{role}:{msg.content}")
    
'''
story is deprecated. Use LangGraph's built-in persistence instead.
  basic_memory()
ai回答Hi Paulo! How can I assist you today?
ai回答That’s great! LangChain is an interesting framework designed to simplify the development of applications using large language models. It provides tools for connecting language models with various data sources, memory, and even external APIs. What aspects of LangChain are you particularly interested in?
ai回答Your name is Paulo, and you are learning about LangChain. If you have any specific questions or topics you'd like to explore, feel free to ask!
打印当前数据库里存的历史信息
human:Hi! My name is Paulo.
ai:Hi Paulo! How can I assist you today?
human:I'm learning about LangChain.
ai:That’s great! LangChain is an interesting framework designed to simplify the development of applications using large language models. It provides tools for connecting language models with various data sources, memory, and even external APIs. What aspects of LangChain are you particularly interested in?
human:What's my name and what am I learning?
ai:Your name is Paulo, and you are learning about LangChain. If you have any specific questions or topics you'd like to explore, feel free to ask!
'''



if __name__ == "__main__":
    basic_memory()

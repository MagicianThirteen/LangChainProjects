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

def message_trimming():
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(
            content="Python is a high-level programming language known for readability and versatility. It's used in web development, data science, AI, and automation."
        ),
        HumanMessage(content="How do I install it?"),
        AIMessage(
            content="You can install Python from python.org or use package managers like apt, brew, or chocolatey. I recommend Python 3.12+ for new projects."
        ),
        HumanMessage(content="What about pip?"),
        AIMessage(
            content="Pip is Python's package installer. It comes with Python 3.4+. Use 'pip install package_name' to install packages. Consider using virtual environments with venv or uv."
        ),
        HumanMessage(content="Can you summarize everything we discussed?"),
    ]

    print(f"原始消息长度{len(messages)}")

    trimed=trim_messages(
        messages=messages,
        token_counter=llm,
        strategy="last",
        allow_partial=False,
        include_system=True,
        max_tokens=60
    )

    print(f"消减后的消息长度{len(trimed)}")
    for msg in trimed:
        role=type(msg).__name__.replace("Message","")
        print(f"{role}:{msg.content}")

    '''
    原始消息长度8
消减后的消息长度2
System:You are a helpful coding assistant.
Human:Can you summarize everything we discussed?
    '''


if __name__ == "__main__":
    #basic_memory()
    message_trimming()

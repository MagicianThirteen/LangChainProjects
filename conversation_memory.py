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

def window_memory():
    #这些是模拟告诉ai的信息
    exchanges = [
        "My name is Paulo",
        "I live in Seattle",
        "I work as an AI engineer",
        "I have 2 cats",
        "What do you remember about me?",
    ]

    class WindowMessageHistory(InMemoryChatMessageHistory):
        k:int=3 #这个要放到类属性里才能被创建的时候就访问
        def add_messages(self, message):
            #定义k，只保留最后k轮的对话
            #self.k=3
            
            super().add_messages(message)
            if len(self.messages)>self.k*2:
                self.messages=self.messages[-(self.k*2):]
            return self.messages
    store:Dict[str,WindowMessageHistory]={}
    def get_session_id(session_id:str):
        if session_id not in store:
            store[session_id]=WindowMessageHistory(k=2)
        return store[session_id]
    
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system","你是个很有用的助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human","{input}")
            
        ]
    )
    chain=prompt|llm|parser
    chain_with_window=RunnableWithMessageHistory(
        chain,
        input_messages_key="input",
        history_messages_key="history",
        get_session_history=get_session_id,
    )

    for q in exchanges:
        print(f"问题是{q}")
        result=chain_with_window.invoke(
            {"input":q},
            config={"configurable":{"session_id":"window_agent"}}
        )
        print(f"回答是{result}")
    
    history=store["window_agent"].messages
    print(f"[Window:{len(history)} msgs]",end="")
    facts_in_memory=[
        m.content[:40] for m in history if isinstance(m,HumanMessage)
    ]
    print(f"Remebers:{facts_in_memory}")

    '''
      window_memory()
问题是My name is Paulo
回答是Nice to meet you, Paulo! How can I assist you today?
问题是I live in Seattle
回答是That's great! Seattle is known for its scenic views, vibrant culture, and coffee scene. Do you have any favorite spots or activities in the city?
问题是I work as an AI engineer
回答是That's intriguing! As an AI engineer, you must be involved in some exciting projects. What specific areas of AI do you work in, or what kind of projects are you currently focusing on?
问题是I have 2 cats
回答是That sounds lovely! Cats can be great companions. What are their names, and what do you enjoy most about having them?
问题是What do you remember about me?
回答是I remember that you work as an AI engineer and that you have two cats. If there's anything specific you’d like me to focus on or if you have more questions, feel free to let me know!
[Window:4 msgs]Remebers:['I have 2 cats', 'What do you remember about me?']
    '''

def summary_memory():
    chat_llm=init_chat_model(model="gpt-4o-mini", temperature=0.7)
    summary_llm=init_chat_model(model="gpt-4o-mini", temperature=0)

    chat_prompt=ChatPromptTemplate.from_messages(
        [("system","You are a helpful assistant. Be concise.\n\nSummary of earlier conversation:\n{summary}"),
        MessagesPlaceholder(variable_name="recent_message"),
        ("human","{input}"),]
    )
    chat_chain=chat_prompt|chat_llm|parser
    summarize_prompt=ChatPromptTemplate.from_template(
        "Condense the current summary and new messages into a single updated summary "
        "(2-3 sentences). Preserve all key facts about the user.\n\n"
        "Current summary:\n{current_summary}\n\n"
        "New messages:\n{new_messages}\n\n"
        "Updated summary:"
    )
    summarize_chain=summarize_prompt|summary_llm|parser

    running_summary=""
    recent_messages=[]
    MAX_RECENT=4

    exchanges = [
        "My name is Paulo and I'm from Seattle",
        "I work as an AI engineer building RAG systems",
        "I have 2 cats named Luna and Milo",
        "I'm building a LangChain course for Udemy",
        "What do you know about me? List everything.",
    ]

    print(f"\nConfig: keep last {MAX_RECENT} messages, summarize the rest\n")

    for user_input in exchanges:
        print(f"User:{user_input}")
        response=chat_chain.invoke(
            {
                "summary":(running_summary if running_summary else "No prior conversation."),
                "recent_message":recent_messages,
                "input":user_input
            }
        )
        print(f"AI:{response}")

        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        if len(recent_messages)>MAX_RECENT:
            messages_to_summarize=recent_messages[:-MAX_RECENT]
            formatted="\n".join(
                f"{'Human' if isinstance(msg,HumanMessage) else 'AI'}:{msg.content}"
                for msg in messages_to_summarize
            )

            running_summary=summarize_chain.invoke(
                {
                    "current_summary":(
                        running_summary if running_summary else "None yet"
                    ),
                    "new_messages":formatted,
                }
            )

            recent_messages=recent_messages[-MAX_RECENT:]

            print(
                f"  >>> Summarized! Compressed {len(messages_to_summarize)} old messages"
            )
            print(f"  >>> Summary: {running_summary}")
            print(f"  >>> Recent buffer: {len(recent_messages)} messages")
        print()

    # --- Final state ---
    print("=" * 60)
    print("FINAL MEMORY STATE")
    print("=" * 60)
    print(f"\nRunning summary (compressed old context):\n  {running_summary}")
    print(f"\nRecent messages kept verbatim ({len(recent_messages)}):")
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")
    print("\nKey insight: ALL facts preserved (name, city, job, cats, course)")
    print("But token cost stays bounded -- old messages are compressed, not deleted!")

    '''
    
Config: keep last 4 messages, summarize the rest

User:My name is Paulo and I'm from Seattle
AI:Nice to meet you, Paulo! How can I assist you today?

User:I work as an AI engineer building RAG systems
AI:That's interesting, Paulo! RAG (Retrieval-Augmented Generation) systems are a fascinating area of AI. What specific aspects of RAG systems are you working on?

User:I have 2 cats named Luna and Milo
AI:They sound adorable! What are Luna and Milo like?
  >>> Summarized! Compressed 2 old messages
  >>> Summary: Paulo is from Seattle and is seeking assistance.
  >>> Recent buffer: 4 messages

User:I'm building a LangChain course for Udemy
AI:That sounds great! LangChain is a powerful framework for building applications with language models. What topics do you plan to cover in your course?
  >>> Summarized! Compressed 2 old messages
  >>> Summary: Paulo, an AI engineer from Seattle, is seeking assistance and specializes in building Retrieval-Augmented Generation (RAG) systems.
  >>> Recent buffer: 4 messages

User:What do you know about me? List everything.
AI:Based on our conversation, I know the following about you:

1. Your name is Paulo.
2. You are an AI engineer from Seattle.
3. You specialize in building Retrieval-Augmented Generation (RAG) systems.
4. You have two cats named Luna and Milo.
5. You are building a LangChain course for Udemy.

If there's anything else you want to share or discuss, feel free!
  >>> Summarized! Compressed 2 old messages
  >>> Summary: Paulo, an AI engineer from Seattle specializing in building Retrieval-Augmented Generation (RAG) systems, has two cats named Luna and Milo.
  >>> Recent buffer: 4 messages

============================================================
FINAL MEMORY STATE
============================================================

Running summary (compressed old context):
  Paulo, an AI engineer from Seattle specializing in building Retrieval-Augmented Generation (RAG) systems, has two cats named Luna and Milo.

Recent messages kept verbatim (4):
  Human: I'm building a LangChain course for Udemy
  AI: That sounds great! LangChain is a powerful framework for building applications w
  Human: What do you know about me? List everything.
  AI: Based on our conversation, I know the following about you:

1. Your name is Paul

Key insight: ALL facts preserved (name, city, job, cats, course)
But token cost stays bounded -- old messages are compressed, not deleted!
    '''
      


if __name__ == "__main__":
    #basic_memory()
    #message_trimming()
    #window_memory()
    summary_memory()

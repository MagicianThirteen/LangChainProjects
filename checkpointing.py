from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

class chatstate(TypedDict):
    messages:Annotated[list[BaseMessage],operator.add]


def memory_saver():
    #只有checkpoint的情况
    def chat(state:chatstate)->dict:
        response=llm.invoke(state["messages"])
        return{
            "messages":[response]
        }
    
    graph=StateGraph(chatstate)
    graph.add_node("chat",chat)
    graph.add_edge(START,"chat")
    graph.add_edge("chat",END)

    agent=graph.compile(checkpointer=MemorySaver())
    config={"configurable":{"thread_id":"demo_1"}}

    result=agent.invoke({"messages":[HumanMessage(content="我叫十三")]},config)
    result=agent.invoke({"messages":[HumanMessage(content="我叫什么？")]},config)
    
    state=agent.get_state(config)
    for m in state.values['messages']:
        print(m)
    '''
    content='我叫十三' additional_kwargs={} response_metadata={}
    content='你好，十三！很高兴认识你。有什么我可以帮助你的吗？' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 10, 'total_tokens': 28, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'latency_checkpoint': {'engine_tbt_ms': 18, 'engine_ttft_ms': 34, 'engine_ttlt_ms': 367, 'pre_inference_ms': 96, 'service_tbt_ms': 19, 'service_ttft_ms': 458, 'service_ttlt_ms': 787, 'total_duration_ms': 701, 'user_visible_ttft_ms': 362}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_965c8b9ecf', 'id': 'chatcmpl-E3YeNGgZKHXqbtSfzOao0XK6TJnik', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019f7d7e-b72a-7d52-80df-60609d479e77-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 10, 'output_tokens': 18, 'total_tokens': 28, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    content='我叫什么？' additional_kwargs={} response_metadata={}
    content='你叫十三。有什么特别的含义吗？' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 12, 'prompt_tokens': 38, 'total_tokens': 50, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'latency_checkpoint': {'engine_tbt_ms': 10, 'engine_ttft_ms': 33, 'engine_ttlt_ms': 147, 'pre_inference_ms': 143, 'service_tbt_ms': 11, 'service_ttft_ms': 517, 'service_ttlt_ms': 644, 'total_duration_ms': 505, 'user_visible_ttft_ms': 373}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_965c8b9ecf', 'id': 'chatcmpl-E3YePe61E9jTMBoiwaM8lHXOInZBi', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019f7d7e-c34c-7951-ac41-cfb7b5537e26-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 38, 'output_tokens': 12, 'total_tokens': 50, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    '''

def sqlite_persistence():
    def chat(state:chatstate)->dict:
        response=llm.invoke(state['messages'])
        return {"messages":[response]}
    
    graph = StateGraph(chatstate)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"\nSQLite Persistence Demo:")
    print(f"Database: {db_path}\n")

    with SqliteSaver.from_conn_string(db_path) as saver:
        agent=graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Remember: The secret code is ALPHA-7")
                ]
            },
            config,
        )
        print(f"Session 1 - Stored secret code")
    
    try:
        with SqliteSaver.from_conn_string(db_path) as saver:
            app = graph.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "persistent-user"}}

            result = app.invoke(
                {"messages": [HumanMessage(content="What was the secret code?")]}, config
            )
            print(f"Session 2 - AI: {result['messages'][-1].content}")  
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
        print(f"已删除临时数据库：{db_path}")

'''

SQLite Persistence Demo:
Database: C:\Users\MAGICI~1\AppData\Local\Temp\tmpfkj5qp3s.db

Session 1 - Stored secret code
Session 2 - AI: The secret code is ALPHA-7.
已删除临时数据库：C:\Users\MAGICI~1\AppData\Local\Temp\tmpfkj5qp3s.db
'''



if __name__ == "__main__":
    #memory_saver()
    sqlite_persistence()

    



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



def checkpointer_branch():
    def chat(state:chatstate)->dict:
        response=llm.invoke(state['messages'])
        return {"messages":[response]}
    
    graph = StateGraph(chatstate)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    agent=graph.compile(checkpointer=MemorySaver())
    main_config={"configurable":{"thread_id":"main"}}
    agent.invoke({"messages":[HumanMessage(content="今天天气如何？")]},main_config)
    main_state=agent.get_state(main_config)

    #分支一，去海边
    b_sea_config={"configurable":{"thread_id":"b_sea"}}
    agent.update_state(b_sea_config,main_state.values)
    result=agent.invoke({"messages":[HumanMessage(content="去海边如何")]},b_sea_config)
    print(f"b_sea: {result['messages'][-1]}")
    #分支二，去爬山
    b_hiking_config={"configurable":{"thread_id":"b_hiking"}}
    agent.update_state(b_hiking_config,main_state.values)
    result=agent.invoke({"messages":[HumanMessage(content="去爬山如何")]},b_hiking_config)
    print(f"b_hiking: {result['messages'][-1]}")
    #这里如果要打印内容要用content
    #print(f"Branch B (Mountain): {result_b['messages'][-1].content[:100]}...")
    '''
    b_sea: content='去海边是个很好的选择！你可以享受阳光、沙滩和海浪，放松心情。以下是一些建议，帮助你更好地享受海边时光：\n\n1. **准备物品**：带上防晒霜、泳衣、毛巾、沙滩椅、遮阳伞和饮用水等。\n\n2. **活动选择**：可以游泳、冲浪、沙滩排球、捡贝壳，或者只是躺在沙滩上晒太阳。\n\n3. **注意安全**：在游泳时要注意海浪和潮汐，确保在安全区域内活动。\n\n4. **环保意识**：保持海滩清洁，不随意丢弃垃圾，保护海洋环境。\n\n5. **享受美食**：如果有机会，可以尝试海边的海鲜美食，或者带上自己喜欢的食物。\n\n希望你在海边度过愉快的时光！' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 221, 'prompt_tokens': 55, 'total_tokens': 276, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'latency_checkpoint': {'engine_tbt_ms': 13, 'engine_ttft_ms': 78, 'engine_ttlt_ms': 2947, 'pre_inference_ms': 106, 'service_tbt_ms': 13, 'service_ttft_ms': 242, 'service_ttlt_ms': 3109, 'total_duration_ms': 3014, 'user_visible_ttft_ms': 136}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_965c8b9ecf', 'id': 'chatcmpl-E3hZu65YAASDUz0caz2CbTWF3fQ03', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019f7f8a-67f1-7863-bf37-ede41aa1567d-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 55, 'output_tokens': 221, 'total_tokens': 276, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}b_hiking: content='爬山是一项很好的户外活动，可以锻炼身体、放松心情，享受大自然的美景。以下是一些建议，帮助你更好地进行爬山活动：\n\n1. **选择合适的山峰**：根据自己的体力和经验选择适合的山峰，初学者可以选择较为平缓的路线。\n\n2. **准备装备**：\n   - **鞋子**：穿着舒适的登山鞋，确保有良好的抓地力。\n   - **衣物**：根据天气情况选择合适的服装，建议穿透气、快干的衣物。\n   - **背包**：准备一个轻便的背包，装上必要的物品。\n\n3. **携带水和食物**：保持水分补充，带一些能量食品，如坚果、能量棒等。\n\n4. **注意安全**：了解山路的情况，遵循登山的安全规则，避免单独行动。\n\n5. **保持节奏**：根据自己的体力调整爬山的节奏，适时休息，享受沿途的风景。\n\n6. **注意环保**：遵循“无痕山林”的原则，不打扰自然环境，带走自己的垃圾。\n\n希望你能享受爬山的乐趣！如果有其他问题，欢迎随时问我。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 304, 'prompt_tokens': 56, 'total_tokens': 360, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'latency_checkpoint': {'engine_tbt_ms': 12, 'engine_ttft_ms': 32, 'engine_ttlt_ms': 3701, 'pre_inference_ms': 331, 'service_tbt_ms': 12, 'service_ttft_ms': 1003, 'service_ttlt_ms': 4622, 'total_duration_ms': 4300, 'user_visible_ttft_ms': 672}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_965c8b9ecf', 'id': 'chatcmpl-E3hZyi1abc71KGdMxQHLTPgqtbPfF', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019f7f8a-76f7-7323-a198-4e2b621f8883-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 56, 'output_tokens': 304, 'total_tokens': 360, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    '''




if __name__ == "__main__":
    #memory_saver()
    #sqlite_persistence()
    checkpointer_branch()

    



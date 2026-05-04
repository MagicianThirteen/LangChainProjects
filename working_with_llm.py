

from dotenv import load_dotenv
from langchain import messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
load_dotenv()
#用新的生成model的方式，通过message设计一个多段对话的demo
def demo_multi_turn_conversation():
    model=init_chat_model(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
        max_retries=3,
        max_tokens=1000,
    )
    sys_msg=SystemMessage(content="你是剑来动画的热心观众")
    hum_msg=HumanMessage(content="一句话形容宁瑶")
    msgs=[sys_msg,hum_msg]
    response=model.invoke(msgs)
    print(f"剑来解释: {response.content}") #为什么这里不用parser了？因为现在的model已经内置了对message的解析能力，直接返回content就好了，不需要再用parser去解析了
    msgs.append(response) 
    msgs.append(HumanMessage(content="一句话再形容一下宁瑶的境界"))
    print("继续对话中...")
    response=model.invoke(msgs)
    print(f"剑来解释: {response.content}")

if __name__=="__main__":
    demo_multi_turn_conversation()
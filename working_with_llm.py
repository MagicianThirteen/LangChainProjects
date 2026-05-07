

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

#通过调用多个模型来检查不同模型的表现
def demo_multis_model_comparison():
    #方案一：
    #定义多个模型，通过init_chat_model
    model1=init_chat_model(model="gpt-4o-mini",temperature=0.7,streaming=True,max_retries=3,max_tokens=1000)
    model2=init_chat_model(model="gpt-4o",temperature=0.7,streaming=True,max_retries=3,max_tokens=1000)
    #模型放进一个列表然后遍历invoke，对比出答案
    models=[model1,model2]
    sys_msg=SystemMessage(content="你是剑来的热心观众")
    hum_msg=HumanMessage(content="用一句话形容宁瑶")
    msgs=[sys_msg,hum_msg]
    for model in models:
        response=model.invoke(msgs)
        print(f"{model.model_name}的关于宁瑶的回答: {response.content}")
        
if __name__=="__main__":
    #demo_multi_turn_conversation()
    demo_multis_model_comparison()

from email import message

from dotenv import load_dotenv
load_dotenv()
#常见通用模板
#基础模板，简单问答,把一段文字翻译成对应的语言{text}{language}
#输出结果：“宁瑶真棒”可以翻译为 “Ning Yao is really awesome” 或者 “Ning Yao is great.”
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain.chat_models import init_chat_model
def basic_prompt_template():
    prompt=ChatPromptTemplate.from_template("把{text}翻译成{language}")
    message=prompt.format_messages(text="宁瑶真棒",language="英文")
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    response=model.invoke(message)
    print(response.content)

#传入多条消息的，定位大模型的角色以及要做的事情，打印传出每种消息的内容是啥
#输出结果：SystemMessage:你是个翻译者，回答要简洁
#HumanMessage:翻译宁瑶真棒成英文
#Ning Yao is awesome.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
def multi_message_template():
    prompt=ChatPromptTemplate.from_messages(
        [("system","你是一个翻译，回答要简洁"),
         ("human","翻译{text}成{language}")
        ]
    )
    #注意，如果写成systemMessage(content="你是一个翻译")
    #            humanMessage(content="翻译{text}成{language}")
    #这样的写法，即使里面有占位符都不会识别，因为只会认成字符串
    message=prompt.format_messages(text="宁瑶真棒",language="英文")
    for msg in message:
        print(f"{type(msg).__name__}:{msg.content}")
    response=init_chat_model(model="gpt-4o-mini",temperature=0).invoke(message)
    print(response.content)


#各种消息类型
#对话结果：35
def message_types_demo():
    message=[
        SystemMessage(content="你是一个数学老师，回答要简洁"),
        HumanMessage(content="5乘以5等于多少？"),
        AIMessage(content="25"),
        HumanMessage(content="如果我再加10呢？")
    ]
    #注意这里的每一条message都是一条完整的消息，不能用占位符
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    response=model.invoke(message)
    print(f"对话结果：{response.content}")


#聊天占位符，用来在多条消息中，占用一段对话
#对话结果：对话结果：在《剑来》中，宁瑶是一个非常重要的女剑仙角色。她不仅实力强大，而且在故事中扮演了关键的角色。宁瑶的性格和经历也为故事增添了许多深度和情感。除了宁瑶，书中还有其他一些女剑仙角色，但宁瑶是最为突出的之一。
def demo_messages_placeholder():
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system","你是一个聪明的助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human","{input}")
        ]
    )

    history=[
        HumanMessage(content="剑来里的女主叫宁瑶，是个剑仙"),
        AIMessage(content="宁剑仙厉害"),
    ]
    message=prompt.format_messages(history=history,input="剑来里很厉害的女剑仙是谁")
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    #response=model.invoke(message
    chain=prompt|model
    response=chain.invoke({"history":history,"input":"剑来里很厉害的女剑仙是谁"})
    #输出结果：对话结果：在《剑来》中，宁瑶是一个非常重要的女剑仙角色，她是宁氏一脉的传人，实力强大，性格坚韧。她在故事中与主角陈平安有着深厚的情感纠葛。宁瑶的剑术和修为在整个故事中都占有重要地位，是一位备受瞩目的女剑仙。
    print(f"对话结果：{response.content}")
    #为什么这里不用model.invoke(message)
    #因为，chain.invoke相当于message = prompt.format_messages(...)
    #response = model.invoke(message)，后面越来越复杂一直prompt.format_messages再model.invoke很繁琐
    
    
    












if __name__ == "__main__":
    #basic_prompt_template()
    #multi_message_template()
    #message_types_demo()
    demo_messages_placeholder()
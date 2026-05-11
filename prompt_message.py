from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage,ChatMessage
from langchain_core.prompts import FewShotChatMessagePromptTemplate

load_dotenv()

#prompt是由变量组成的字符串，而且字符串可以被赋值定义，适用场合
def ChatPromptTemplateBasicDemo():
    prompt=ChatPromptTemplate.from_template("告诉我这个{book}的某个人物，比如{Characters}的一条信息")
    message=prompt.format_messages(book="剑来", Characters="宁瑶")
    print(message)

#通过传入多条信息来定义prompt
def MultiMessagePromptDemo():
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system","你是个翻译助手，负责将{input_language}翻译成{output_language}"),
            ("user","请翻译这句话：{text}")
        ]
    )
    message=prompt.format_messages(input_language="中文",output_language="英文",text="宁瑶姑娘太酷了")
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    response=model.invoke(message)
    print(response.content)
    #为什么这里不用parser呢？

#定义各种不同的message
def MultiKindMessage():
    message=[
    SystemMessage(content="你是个剑来的热心读者"),
    HumanMessage(content="请用一句话介绍一下宁瑶姑娘"),
    AIMessage(content="好的，我调用工具搜索一下宁瑶姑娘的相关信息"),
    ToolMessage(content="正在搜索中..."),   
    ]

#给ai一个小样本，让它模仿这个东西来生成
#比如输入一个词的，返回一个词的反义词
def FewShotChatMessagePromptDemo():
    #exemples,这里是个list[dict]
    exemples=[
        {"input":"happy","output":"sad"},
        {"input":"hot","output":"cold"}
    ]
    #prompts,告诉大模型，用户输入什么，ai输出什么
    exemples_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )
    #拼成最终提示模板
    final_prompt=FewShotChatMessagePromptTemplate(
    example_prompt=exemples_prompt,
    examples=exemples

    )
    message=ChatPromptTemplate.from_messages(
        [
            ("system","请给出每个词的反义词"),
            final_prompt,
            ("human","{input}")
        ]
    )
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    response=model.invoke(message.format_messages(input="tall"))
    print(response.content)
    #model调用




if __name__ == "__main__":
    #ChatPromptTemplateBasicDemo()
    #MultiMessagePromptDemo()
    FewShotChatMessagePromptDemo()
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

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

if __name__ == "__main__":
    #ChatPromptTemplateBasicDemo()
    MultiMessagePromptDemo()
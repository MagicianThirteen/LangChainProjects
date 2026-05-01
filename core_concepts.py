from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

#简单的展示下langchain的核心概念，runnables和lecl语言
def langchain_core_concepts():
    #这里实现一个简单问答，问openai，langchain的一个解释
    prompt=ChatPromptTemplate.from_template("你是个ai技术专家，一句话解释下{question}")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
    parser=StrOutputParser()
    chain=prompt|model|parser #这里不能随便改顺序，model会不知道怎么输出
    result=chain.invoke({"question":"什么是langchain？"})
    print(f"结果：{result}")
    return chain
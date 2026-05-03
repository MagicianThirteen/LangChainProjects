from itertools import product

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

#通过batch处理一批输入
def batch_demo():
    prompt=ChatPromptTemplate.from_template("翻译成英文：{text}")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.2)
    parser=StrOutputParser()
    chain=prompt|model|parser
    inputs=[{"text":"我是谁"},{"text":"我在干什么"}]
    results=chain.batch(inputs)
    for input_item,output_item in zip(inputs,results):
        print(f"输入：{input_item['text']}=>输出：{output_item}")

#通过stream()表现实时输出
def streaming_demo():
    prompt=ChatPromptTemplate.from_template("写一句关于{topic}的台词")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.8)
    parser=StrOutputParser()
    chain=prompt|model|parser
    for chunk in chain.stream({"topic":"剑来"}):
        print(chunk,end="",flush=True) #end=""表示不换行，flush=True表示立即输出，不缓冲
    print() #换行

#检查内部的输入输出格式
def schema_inspection_demo():
    prompt=ChatPromptTemplate.from_template("总结一下文本：{text}")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
    parser=StrOutputParser()
    chain=prompt|model|parser
    #chain.invoke({"text":"剑来是部国产漫画，里面有很帅的女主角宁瑶"})

    input_schema=chain.input_schema.model_json_schema()
    output_schema=chain.output_schema.model_json_schema()
    print(f"输入格式：{input_schema}")
    print(f"输出格式：{output_schema}")

#build一个简单的chain，生成一个产品的营销标语
def build_production_chain():
    prompt=ChatPromptTemplate.from_template("为定位是{audience}的人群定制一条{product}的营销标语，要求简短有效")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
    parser=StrOutputParser()
    chain=prompt|model|parser
    result=chain.invoke({"product":"人工智能课程","audience":"开发者"})
    print(f"营销标语：{result}")
    #输出营销标语："解锁AI潜能，成就开发者未来！"
    


if __name__ == "__main__":
    #langchain_core_concepts()
    #batch_demo()
    #streaming_demo()
    #schema_inspection_demo()
    build_production_chain()
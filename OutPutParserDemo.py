from dotenv import load_dotenv
load_dotenv()

#stroutputparser,用来简单的问答提取需要的字符串
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

#输出这个<class 'langchain_core.messages.base.TextAccessor'>
#宁瑶是小说《剑来》中的重要角色之一。她是宁家的一名修士，性格聪慧、果断，具有很强的个人魅力和领导能力。宁瑶在故事中不仅展现了出色的修炼天赋，还在关键时刻展现了她的智慧和勇气。她与主角陈平安之间的关系复杂，既有合作也有冲突，推动了故事的发展。

#宁瑶的角色体现了女性在修真世界中的独立与坚韧，同时也反映了人性中的情感与道德抉择。她的经历和成长为整个故事增添了深度和层次。
#换英文测试，中文好贵
def StrOutPutParserTest():
    #prompt
    prompt=ChatPromptTemplate.from_template("简单解释下剑来中的{name}")
    #model
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    #stroutputparser
    strparser=StrOutputParser()
    chain=prompt|model|strparser
    response=chain.invoke({"name":"宁瑶"})
    print(type(response))#<class 'langchain_core.messages.base.TextAccessor'>
    print(response)


#使用jsonparser，把用户的描述，拆成可以被处理的结构数据
#输出结果：{'name': 'Alex', 'age': 25}
from langchain_core.output_parsers import JsonOutputParser
def JsonOutPutParserTest():
    #prompt
    prompt=ChatPromptTemplate.from_template("Return a JSON object with 'name' and 'age' for: {description} ")
    #model
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    #jsonOutPutParser
    jsonOutPutParser=JsonOutputParser()
    chain=prompt|model|jsonOutPutParser
    response=chain.invoke({"description":"A 25-year-old developer named Alex"})
    print(response)





if __name__ == "__main__":
    #StrOutPutParserTest()\
    JsonOutPutParserTest()
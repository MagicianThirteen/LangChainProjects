#几种chain
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

#最基本的chain
def basic_chain_demo():
    #prompt
    prompt=ChatPromptTemplate.from_template("{book}的女主角是谁,只用说名字")
    #model
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    #strparser
    parser=StrOutputParser()
    chain=prompt|model|parser
    result=chain.invoke({"book":"剑来"})
    print(f"结果：{result}")

#用来控制并行的chain
def parallel_chain_demo():
    #定义几个可以并行执行的prompt，比如一个回答，一个用来一句话描写
    answer_prompt=ChatPromptTemplate.from_template("{book}的女主角是谁,只用说名字")
    description_prompt=ChatPromptTemplate.from_template("用一句话描写{book}的女主角")
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    parser=StrOutputParser()
    parallel_chain=RunnableParallel(
        answer=answer_prompt|model|parser,
        description=description_prompt|model|parser,
    )
    result=parallel_chain.invoke({"book":"剑来"})
    print(f"并行结果：{result['answer']}")
    print(f"并行结果：{result['description']}")






if __name__ == "__main__":
    #basic_chain_demo()
    parallel_chain_demo()
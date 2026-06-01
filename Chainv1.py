#几种chain
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,)

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


#passthrough chain,把最初输入传下去
def passthrough_chain_demo():
    prompt=ChatPromptTemplate.from_template(
        "请根据上下文{context}回答问题{question}"
    )
    #定义一个查找返回的假数据的函数
    def fake_retriever(input_dic):
        return "剑来的女主是宁瑶"
    #定义并行处理的runnable
    parallel=RunnableParallel(
        context=RunnableLambda(fake_retriever),
        question=RunnablePassthrough(),
    )
    #定义一个lambda来整理数据
    funx=RunnableLambda(
        lambda x:{"context":x["context"],"question":x["question"]["question"]}
    )
   
    #定义一个model
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    #定义一个解析strparser
    parser=StrOutputParser()
    chain=parallel|funx|prompt|model|parser
    result=chain.invoke({"question":"剑来的女主是谁?"})
    print(f"passthrough结果：{result}")

#Branching demo，更具输入的不同问题选择对应的agent
#比如回答是关于剑来书里的角色问题，还是关于剑来的作者问题
#输出结果：branching结果：《剑来》的女主角是李清照。她是小说中的重要角色之一，性格坚韧，聪慧过人，与男主角陈平安之间有着复杂的情感纠葛。小说通过她的视角展现了许多情节和人物关系。
#branching结果：《剑来》的作者是烽火戏诸侯。这部小说在网络上非常受欢迎，讲述了一个关于修仙和江湖的故事。
def branching_chain_demo():
    #定义通用model
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    #定义通用parser
    parser=StrOutputParser()
    #定义一个角色问题的，prompt
    role_prompt=ChatPromptTemplate.from_template("{book}的{question}")
    #定义一个作者问题的，prompt
    author_prompt=ChatPromptTemplate.from_template("{book}的{question}")
    #定义一个用来分类的，prompt
    classifier_prompt=ChatPromptTemplate.from_template("{question}是关于角色问题还是作者问题?只用回答角色或者作者")
    classifier=classifier_prompt|model|parser
    #定义一个根据分类prompt返回值来确定哪个分支的函数
    def is_role_question(input_dic):
        classification=classifier.invoke(input_dic)
        return "角色" in classification
    #定义一组question来测试，看看输出如何
    questions=[
        "剑来的女主是谁?",
        "剑来的作者是谁?"
    ]

    branch=RunnableBranch(
            (is_role_question,role_prompt|model|parser),
            author_prompt|model|parser,
        )

    for q in questions: 
        result=branch.invoke({"book":"剑来","question":q})
        print(f"branching结果：{result}")
        
       
    


#debug chain
def demo_debbuging():
    prompt=ChatPromptTemplate.from_template("{book}"
    "的女主角的名字")
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    parser=StrOutputParser()
    #chain=prompt|model|parser
    #chain.invoke({"book":"剑来"})
    #检查输入，输出的内部结构
    #print("内部输入结构：",chain.input_schema.model_json_schema())
    #print("输出结构: ",chain.output_schema.model_json_schema())
    #输出结果：
    #内部输入结构： {'properties': {'book': {'title': 'Book', 'type': 'string'}}, 'required': ['book'], 'title': 'PromptInput', 'type': 'object'}
    #输出结构:  {'title': 'StrOutputParserOutput', 'type': 'string'}
    
    #方便langsmith查找调用情况，用with_config
    # result=chain.with_config(
    #     run_name="book_chain"
    # ).invoke({"book":"剑来"})
    #print(f"book_chain:{result}")
    #输出：book_chain:《剑来》的女主角是李青莲。
    #这里的run_name影响的是langsmith看到的东西
    #通过插入runnablelambda来测试log，打印两个阶段的东西
    #after prompt after model
    def log_step(x,step_name):
        print(f"[{step_name}]:{type(x).__name__}:{str(x)[:10]}")
        return x
    chain=(prompt|
           RunnableLambda(lambda x:log_step(x,"after prompt"))
           |model|
           RunnableLambda(lambda x:log_step(x,"after model"))
           |parser
           )
    result=chain.invoke({"book":"剑来"})
#     输出这个
#     [after prompt]:ChatPromptValue:messages=[
#     [after model]:AIMessage:content='《
#     book_chain:《剑来》的女主角是李清照。她是小说中的重要角色之一，具有独特的个性和背景。小说围绕她与主角的互动以及他们在修仙世界中的冒险展开。
    print(f"book_chain:{result}")
    
        




if __name__ == "__main__":
    #basic_chain_demo()
    #parallel_chain_demo()
    #passthrough_chain_demo()
    #branching_chain_demo()
    demo_debbuging()
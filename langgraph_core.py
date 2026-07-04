
from langgraph.graph import StateGraph,START,END
from typing import Literal, TypedDict,Annotated
import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

#基本状态模型
class SimpleState(TypedDict):
    input:str
    output:str
    step:int

def demo_simple_graph():

    #定义处理节点
    def process(state:SimpleState)->dict:
        return{"output":state["input"].upper(),
               "step":state["step"]+1}
    #graph
    graph=StateGraph(SimpleState)
    #添加节点
    graph.add_node("process",process)
    #边
    graph.add_edge(START,"process")
    graph.add_edge("process",END)

    app=graph.compile()
    #这里的result也返回一个dict
    result=app.invoke({"input":"hello agent",
                       "output":"",
                       "step":0})
    #注意，括号里是单引号！！！
    print(f" Input: {result['input']}, Output: {result['output']}, Step: {result['step']}")

    print(app.get_graph().draw_mermaid())
    png_bytes=app.get_graph().draw_mermaid_png()
    with open("graph.png","wb") as f:
        f.write(png_bytes)

    '''
     Input: hello agent, Output: HELLO AGENT, Step: 1
    '''

def accumulating_state():
    class AccumulatingState(TypedDict):
        messages:Annotated[list[str],operator.add]
        count:Annotated[int,operator.add]
    
    def step_one(state:AccumulatingState)->dict:
        return {"messages":["step 1"],
                "count":1}
    def step_two(state:AccumulatingState)->dict:
        return {"messages":["step 2"],
                "count":1}
    graph=StateGraph(AccumulatingState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)

    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    app=graph.compile()
    # # visualize the graph
    print("\n--- Mermaid Graph ---")
    print(app.get_graph().draw_mermaid())

    # save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_2.png", "wb") as f:
        f.write(png_bytes)

    result = app.invoke({"messages": ["Initial message"], "count": 0})
    print(f"messages:{result['messages']}\n count:{result['count']}")

    '''
    
--- Mermaid Graph ---
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        step_one(step_one)
        step_two(step_two)
        __end__([<p>__end__</p>]):::last
        __start__ --> step_one;
        step_one --> step_two;
        step_two --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc

messages:['Initial message', 'step 1', 'step 2']
 count:2
    
    '''

#==== Message State(Common pattern) ===
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langchain.chat_models import init_chat_model

class MessageState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def message_state():
    llm=init_chat_model("gpt-4o-mini",temperature=0)
    def chat_node(state:MessageState)->dict:
        response=llm.invoke(state['messages'])
        return {"messages":[response]}
    
    graph = StateGraph(MessageState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    #注意，这里llm.invoke返回的是AIMessage
    #agent.invoke，返回的是一个字典，是关于定义好的state的状态的字典
    agent=graph.compile()
    result=agent.invoke({"messages":[HumanMessage(content="剑来的女主叫什么名字")]})
    for msg in result["messages"]:
        role="human" if isinstance(msg,HumanMessage) else "ai"
        print(f"  {role}: {msg.content}")
    
    '''
      human: 剑来的女主叫什么名字
      ai: 《剑来》的女主角叫做“李清照”。她是小说中的重要角色之一，具有独特的个性和背景。小说围绕着她与主角之间的故事展开。如果你对这个角色或小说有更多问题，欢迎提问！
    '''

#练习
def demo_langgraph():
    llm=init_chat_model(model="gpt-4o-mini",temperature=0)
    class QuestionState(TypedDict):
        topic:str
        three_questions:list[str]
        choose_question:str
        answer:str
    class response(BaseModel):
        questions:list[str]=Field(description="用来存放通过主题生成的三个问题")
        choose_question:str=Field(description="用来存放在三个问题当中选择要回答的那个问题")
        question_answer:str=Field(description="用来存放选择的这个问题的答案")
    
    structed_llm=llm.with_structured_output(response)
    def node_1(state:QuestionState)->dict:
        system=f"""
        你是一个很有用的助手，请根据{state["topic"]}
        生成三个与之相关的简短问题
        """
        response=structed_llm.invoke([SystemMessage(content=system)])
        return {
            "three_questions":response.questions
        }
    
    def node_2(state:QuestionState)->dict:
        questions=",".join(state["three_questions"])
        system=f"""
                你是一个很有用的助手，从以下几个问题{questions}中，
                选一个作为你想回答的问题，并对问题的答案做简短回答
        """
        result=structed_llm.invoke([SystemMessage(content=system)])
        return {
            "choose_question":result.choose_question,
            "answer":result.question_answer
        }
    
    graph=StateGraph(QuestionState)

    graph.add_node("node_1",node_1) 
    graph.add_node("node_2",node_2)

    graph.add_edge(START,"node_1")
    graph.add_edge("node_1","node_2")
    graph.add_edge("node_2",END) 

    agent=graph.compile()

    result=agent.invoke({
        "answer":"",
        "choose_question":"",
        "three_questions":[],
        "topic":"langgraph"
    }) 

    for q in result["three_questions"]:
        print(f"选出的问题：{q}")

    print(f"选择的问题是：{result['choose_question']}")
    print(f"问题的回答是：{result['answer']}")  

    '''
    选出的问题：How does LangGraph visualize language models?
    选出的问题：What types of language data can be analyzed using LangGraph?
    选出的问题：In what ways can LangGraph improve language processing tasks?
    选择的问题是：How does LangGraph visualize language models?
    问题的回答是：LangGraph visualizes language models by creating interactive graphs that represent the relationships and structures within the language data, allowing users to explore and analyze the connections between different linguistic elements.
    '''    

class QualityState(TypedDict):
    content:str
    quality_score:int
    feed_back:str
    final_content:str
    iteration:int

llm = init_chat_model("gpt-4o-mini", temperature=0.0)

def condition_loop():
    def evalute(state:QualityState)->dict:
        system=(f"Rate this content quality from 1-10. Reply with just the number.\n\n"
                f"Content: {state['content']}")
        response=llm.invoke(system)
        try:
            score=int(response.content.strip())
        except ValueError:
            score=5
        return{
            "quality_score":score
        }
    
    def improve(state:QualityState)->dict:
        system=f"Improve this content to be more engaging and clear:\n\n{state['content']}"
        response=llm.invoke(system)
        return{
            "content":response.content,
            "iteration":state["iteration"] + 1
        }

    def finalize_content(state:QualityState)->dict:
        return{
            "final_content":state["content"],
            "feed_back":f"Approved after {state['iteration']} iterations with score {state['quality_score']}",
        }
    
    def should_continue(state:QualityState)->Literal["improve","finalize"]:
        if state["quality_score"]>=7:
            return "finalize"
        elif state["iteration"]>=3:
            return "finalize"
        else:
            return "improve"
    
    graph=StateGraph(QualityState)

    graph.add_node("evaluate",evalute)
    graph.add_node("improve", improve)
    graph.add_node("finalize", finalize_content)\
    
    graph.add_edge(START,"evaluate")

    graph.add_conditional_edges("evaluate",
                                should_continue,
                                {"improve":"improve",
                                 "finalize":"finalize"})
    graph.add_edge("improve","evaluate")
    graph.add_edge("finalize",END)

    agent=graph.compile()

    result=agent.invoke(
        {"content":"ai is coool",
         "quality_score":0,
         "feed_back":"",
         "final_content":"",
         "iteration":0}
    )

    print(f"Original: AI is cool")
    print(f"Final: {result['final_content'][:200]}...")
    print(f"Feedback: {result['feed_back']}")


    '''
    Original: AI is cool
    Final: AI is incredibly cool! It’s transforming the way we live, work, and interact with the world around us. From smart assistants that help us manage our daily tasks to advanced algorithms that drive innov...
    Feedback: Approved after 1 iterations with score 7
    '''
    


        



if __name__ == "__main__":  
    #demo_simple_graph()
    #accumulating_state()
    #message_state()
    #demo_langgraph() 
    condition_loop()
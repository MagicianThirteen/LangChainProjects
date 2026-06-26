from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from dotenv import load_dotenv

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

if __name__ == "__main__":  
    demo_simple_graph() 
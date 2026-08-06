from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import asyncio
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

class ParallelState(TypedDict):
    query:str
    research_result:str
    creative_result:str
    #technical_result:str
    final_synthesis:str

def Parallel_Research():
    def research(state:ParallelState)->dict:
        system=f"根据主题{state['query']}做低于20个字的简短调查"
        response=llm.invoke(system)
        #最好写成这样SystemMessage+HumanMessage的方式
        # response = llm.invoke(
        #             [
        #                 SystemMessage(
        #                     content="You are an academic researcher. Provide factual, well-sourced information."
        #                 ),
        #                 HumanMessage(content=f"Research this topic: {state['query']}"),
        #             ]
        #         )
        return{
            "research_result":response.content
        }

    def creative(state:ParallelState)->dict:
        system=f"根据主题{state['query']}做低于20个字的创造性调查"
        response=llm.invoke(system)
        return{
            "creative_result":response.content
        }
    
    #def technical(state:ParallelState)->dict:
        #return

    def synthesis(state:ParallelState)->dict:
        system=f"根据{state['research_result']}和{state['creative_result']}分析，汇总成字数低于30个词的报告"
        response=llm.invoke(system)
        #这里这样写好点，而且好扩展
        #   """Combine all perspectives."""
        #         synthesis_prompt = f"""Synthesize these three perspectives into a comprehensive response:
        
        #         RESEARCH: {state['research_result']}
        
        #         CREATIVE: {state['creative_result']}
        
        #         TECHNICAL: {state['technical_result']}
        
        #         Create a unified, well-structured response."""
        
        #         response = llm.invoke(
        #             [
        #                 SystemMessage(
        #                     content="You are an expert synthesizer. Combine multiple perspectives into coherent insights."
        #                 ),
        #                 HumanMessage(content=synthesis_prompt),
        #             ]
        #         )
        return{
            "final_synthesis":response.content
        }

    graph=StateGraph(ParallelState)
    graph.add_node("research",research)
    graph.add_node("creative",creative)
    #graph.add_node("technical",technical)
    graph.add_node("synthesis",synthesis)

    graph.add_edge(START,"research")
    graph.add_edge(START,"creative")
    #graph.add_edge(START,"technical")

    graph.add_edge("research","synthesis")
    graph.add_edge("creative","synthesis")
    #graph.add_edge("technical","synthesis")

    graph.add_edge("synthesis",END)

    agent=graph.compile()

    result=agent.invoke({
        "query":"关于《剑来》里宁瑶的资料",
        "creative_result":"",
        "final_synthesis":"",
        "research_result":"",
       
    })
    print(f"research结果是{result['research_result']}")
    print(f"creative结果是{result['creative_result']}")
    print(f"汇总结果是{result['final_synthesis']}")

    '''
    research结果是宁瑶是《剑来》中的重要角色，性格坚韧，聪慧机智。
    creative结果是宁瑶：剑术高强，心思细腻，情感复杂，命运多舛。
    汇总结果是宁瑶是《剑来》的重要角色，性格坚韧、聪慧，剑术高强，情感复杂，命运多舛，展现出丰富的人物深度。
    '''


if __name__ == "__main__":
    Parallel_Research()
    
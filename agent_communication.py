from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
import json
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class SharedField(TypedDict):
    query:str
    raw_data:Annotated[list[dict],operator.add]#有source和finding两个字段
    analysis:str
    confidence_score:str
    advisor:list[str]

def AdvisorAgent():
    def datacollector(state:SharedField)->dict:
        #需要通过用户问题，获得依据
        system="""
            "You are a data collector. Given the query, produce 3 data points,每个data point不超过20个词 "
            "as a JSON array of objects with 'source' and 'finding' keys. "
            "Return ONLY the JSON array, no markdown."
        """
        response=llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=state["query"])
            ]
        )
        #这里要应对json转换以及可能转换失败的情况
        try:
            data=json.loads(response.content)
        except json.JSONDecodeError:
            data=[{"source":"gpt-4o","finding":response.content}]

        return{
            "raw_data":data
        }

    def analysis(state:SharedField)->dict:
        system="""
                "You are a data analyst. Analyze the collected data and provide: "
                "1) A brief analysis (1 sentences), and 不超过20个词"
                "2) A confidence score from 0.0 to 1.0. "
                "Format: ANALYSIS: <text>\nCONFIDENCE: <number>"  
        """
        data=json.dumps(state["raw_data"],indent=2)
        human=f"query:{state['query']}\n\nData:\n{data}"
        response=llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=human)
            ]
        )
        #这里最好用pydantic，不用人工去区分
        content=response.content
        if "CONFIDENCE" in content:
            parts=content.split("CONFIDENCE")
            analysis=parts[0].replace("ANALYSIS:","").strip()
            try:
                score=float(parts[1].strip())
            except ValueError:
                score="0.7"

        return{
            "analysis":analysis,
            "confidence_score":score
        }

    def advisor(state:SharedField):
        system="""
            "You are a strategic advisor. Based on the analysis and "
            "confidence score, provide 3 actionable recommendations.每条建议不超过20个词 "
            "Return them as a JSON array of strings. "
            "Return ONLY the JSON array, no markdown."
        
        """
        response=llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=f"query:{state['query']} \nanalysis:{state['analysis']} \nconfidence_score:{state['confidence_score']}")
            ]
        )
        try:
            data=json.loads(response.content)
        except json.JSONDecodeError:
            data=[response.content]
        return{
            "advisor":data
        }

    

    graph=StateGraph(SharedField)
    graph.add_node("data",datacollector)
    graph.add_node("analysis",analysis)
    graph.add_node("advisor",advisor)

    graph.add_edge(START,"data")
    graph.add_edge("data","analysis")
    graph.add_edge("analysis","advisor")
    graph.add_edge("advisor",END)


    agent=graph.compile()

    result=agent.invoke({
        "query":"要不要做agent开发",
        "advisor":[],
        "analysis":"",
        "confidence_score":"",
        "raw_data":[{}]

    })

    for i in result["advisor"]:
        print(i)

if __name__ == "__main__":
    AdvisorAgent()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
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
creative_llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)

class ResearchState(TypedDict):
    topic:str
    messages:Annotated[list[BaseMessage],add_messages]
    search_queries:list[str]
    findings:Annotated[list[dict],operator.add]
    analyst:str
    report:str
    quality_score:float
    quality_feedback:str
    iteration:int

class SearchTaskState(TypedDict):
    search_query:str
    findings:Annotated[list[dict],operator.add]

def multi_agent_research_system():

    def supervisor(state:ResearchState)->dict:
        response=llm.invoke([
             SystemMessage(content="You are a research supervisor. Given a topic, generate exactly 3 "
                            "specific search queries that will cover different angles of the topic. "
                            "Return ONLY a JSON array of strings. No markdown formatting."),
             HumanMessage(content=f"Research topic: {state['topic']}")               
        ])
        try:
            queries=json.loads(response.content)
        except json.JSONDecodeError:
            queries = [
                        f"{state['topic']} overview",
                        f"{state['topic']} latest developments",
                        f"{state['topic']} practical applications",
                    ]

        return{
            "search_queries":queries[:3],
            "messages":[AIMessage(content=f"[SUPERVISOR]: Planned {len(queries)} research queries: {queries}",name="supervisor")]
        }

    def research(state:SearchTaskState)->dict:
        response=llm.invoke([
            SystemMessage(content= "You are a web research agent. For the given search query, "
                        "provide 2-3 key findings. Each finding should have a 'title' "
                        "and 'detail' field. Return a JSON array. No markdown."),
            HumanMessage(content=f"Search query: {state['search_query']}")            
        ])
        try:
            findings=json.loads(response.content)
        except json.JSONDecodeError:
            findings=[{
                "title":state["search_query"],
                "detail":response.content
            }]
        #添加一个tag
        for r in findings:
            r["source_query"]=state["search_query"]

        return{
            "findings":findings
        }

    #用来分发的函数
    def dispatch_searches(state:ResearchState)->list[Send]:
        return[
            Send("research",{
                "search_query":query,
                "findings":[]
            })
            for query in state["search_queries"]
        ]

    def analyst(state:ResearchState)->dict:
        findings=json.dumps(state["findings"],indent=2)
        response=llm.invoke([
            SystemMessage(content="You are a research analyst. Synthesize the collected findings into "
                    "a clear analysis. Identify:\n"
                    "1. Key themes across all findings\n"
                    "2. Any contradictions or gaps\n"
                    "3. The most important insights\n\n"
                    "Write 2-3 paragraphs."),
            HumanMessage(content=
                            f"Research topic: {state['topic']}\n\n"
                            f"Collected findings:\n{findings}")
        ])

        return{
            "analyst":response.content,
            "messages":[AIMessage(content=f"[analyst]: {response.content}",name="analyst")]
        }

    def report_writer(state:ResearchState)->dict:
        findings=json.dumps(state["findings"])
        if state["iteration"]<2:


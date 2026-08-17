
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class TeamState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    query:str
    final_answer:str

def Research_Team()->StateGraph:
    def web_search(state:TeamState)->dict:
        result=llm.invoke(
            [
                SystemMessage(content="You are a web researcher. Find key facts and data about "
                            "the topic. Provide 1-2 bullet points of findings. Be specific.20个词以内"),
                HumanMessage(content=state["query"])
            ]
        )
        return{
            "messages":[AIMessage(content=f"[web_search] {result.content}",name="web_search")]
        }

    def paper_review(state:TeamState)->dict:
        result=llm.invoke(
                    [
                        SystemMessage(content="You are an academic reviewer. Provide technical depth and "
                                                "cite relevant concepts or frameworks. 1-2 bullet points.20个词以内"),
                        HumanMessage(content=state["query"])
                    ]
                )

        return{
                    "messages":[AIMessage(content=f"[paper_review] {result.content}",name="paper_review")]
                }

    def research_lead(state:TeamState)->dict:
        result=llm.invoke([
            SystemMessage(content="You are the research lead. Synthesize the web researcher's "
                            "and paper reviewer's findings into a cohesive research brief. "
                            "Keep it to one short paragraph.大概30个词以内"),
            *state["messages"]
        ])
        return{
            "final_answer":f"research_lead:{result.content}"
        }

    graph=StateGraph(TeamState)
    graph.add_node("web_research",web_search)
    graph.add_node("paper_review",paper_review)
    graph.add_node("research_lead",research_lead)

    graph.add_edge(START,"web_research")
    graph.add_edge(START,"paper_review")
    graph.add_edge("web_research","research_lead")
    graph.add_edge("paper_review","research_lead")
    graph.add_edge("research_lead",END)

    return graph


def test_single_partment():
    testagent=Research_Team().compile()
    result=testagent.invoke({
        "query":"简单解释下rag，不超过20个词以内",
        "messages":[],
        "final_answer":""
        
    })
    print(f"final_answer：{result['final_answer']}")

    '''
    final_answer：research_lead:RAG（Retrieval-Augmented Generation）结合信息检索与生成模型，显著提升文本生成的准确性与相关性，成为自然语言处理领域的重要技术。
    '''


if __name__ == "__main__":
    test_single_partment()

    
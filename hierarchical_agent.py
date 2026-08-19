
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

def build_content_team() -> StateGraph:
    """Build the content department subgraph."""

    def content_writer(state: TeamState) -> dict:
        """Writes content based on available context."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a skilled content writer. Using any research or context "
                        "in the conversation, write a clear, engaging short piece "
                        "(one paragraph). Match a professional but accessible tone.最多20个词"
                    )
                ),
                *state["messages"],
            ]
        )

        return {
            "messages": [
                AIMessage(
                    content=f"[WRITER]: {response.content}", name="content_writer"
                )
            ]
        }

    def content_editor(state: TeamState) -> dict:
        """Edits and polishes the writer's output."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a content editor. Take the writer's draft and "
                        "improve clarity, fix any issues, and tighten the language. "
                        "Return the polished version only.最多20个词"
                    )
                ),
                *state["messages"],
            ]
        )

        return {
            "messages": [
                AIMessage(
                    content=f"[EDITOR]: {response.content}", name="content_editor"
                )
            ],
            "final_answer": response.content,
        }

    content_graph = StateGraph(TeamState)

    content_graph.add_node("writer", content_writer)
    content_graph.add_node("editor", content_editor)

    content_graph.add_edge(START, "writer")
    content_graph.add_edge("writer", "editor")
    content_graph.add_edge("editor", END)

    return content_graph


def build_analysis_team() -> StateGraph:
    """Build the analysis department subgraph."""

    def data_analyst(state: TeamState) -> dict:
        """Provides data-driven analysis."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data analyst. Analyze the topic with numbers, "
                        "trends, and quantitative reasoning. Provide 3-4 data-driven "
                        "insights. Make up plausible stats for demonstration.总共不超过20个词"
                    )
                ),
                *state["messages"],
            ]
        )

        return {
            "messages": [
                AIMessage(
                    content=f"[DATA ANALYST]: {response.content}", name="data_analyst"
                )
            ]
        }

    def strategy_advisor(state: TeamState) -> dict:
        """Provides strategic recommendations."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strategy advisor. Based on the data analysis in the "
                        "conversation, provide 3 actionable strategic recommendations. "
                        "Be specific and practical.总共不超过20个词"
                    )
                ),
                *state["messages"],
            ]
        )

        return {
            "messages": [
                AIMessage(
                    content=f"[STRATEGY ADVISOR]: {response.content}",
                    name="strategy_advisor",
                )
            ],
            "final_answer": response.content,
        }

    analysis_graph = StateGraph(TeamState)

    analysis_graph.add_node("data_analyst", data_analyst)
    analysis_graph.add_node("strategy_advisor", strategy_advisor)

    analysis_graph.add_edge(START, "data_analyst")
    analysis_graph.add_edge("data_analyst", "strategy_advisor")
    analysis_graph.add_edge("strategy_advisor", END)

    return analysis_graph

def build_hierarchical_agent()->StateGraph:
    #构建子图
    research_sub=Research_Team().compile()
    content_sub=build_content_team().compile()
    analysis_sub=build_analysis_team().compile()

    class RouteTeam(BaseModel):
        department:Literal["research", "content", "analysis"]=Field(description="Which department should handle this request")
        reason:str=Field(description="Why this department was chosen")

    def ceo(state:TeamState)->dict:
        system="""
            "You are the CEO supervisor. Route the request to the right department:\n"
                        "- research: Fact-finding, investigation, technical deep-dives\n"
                        "- content: Writing, blog posts, marketing copy, summaries\n"
                        "- analysis: Data analysis, strategy, business decisions\n\n"
                        "Choose the BEST fit department."

        """
        llm_with_structoutput=llm.with_structured_output(RouteTeam)
        response=llm_with_structoutput.invoke([
            SystemMessage(content=system),
            *state["messages"]
        ])
        return{
            "messages":[AIMessage(content=f"[ceo]选这个部门{response.department}，原因是{response.reason}",name="ceo")]

        }

    def route_team(state:TeamState)->Literal["research", "content", "analysis"]:
        #从最新的ceo消息里找ceo的判断的信息
        for msg in reversed(state["messages"]):
            last_ceo_msg=None
            if isinstance(msg,AIMessage) and msg.name=="ceo":
                last_ceo_msg=msg
                break

        if "research" in last_ceo_msg.content.lower():
            return "research"
        if "content" in last_ceo_msg.content.lower():
            return "content"
        if "analysis" in last_ceo_msg.content.lower():
            return "analysis"
        return "research" #default

    parent=StateGraph(TeamState)
    parent.add_node("ceo",ceo)
    parent.add_node("research",research_sub)
    parent.add_node("content",content_sub)
    parent.add_node("analysis",analysis_sub)

    parent.add_edge(START,"ceo")
    parent.add_conditional_edges("ceo",route_team,{
        "research":"research",
        "content":"content",
        "analysis":"analysis"
    })

    return parent


def hierarchical_test():
    agent=build_hierarchical_agent().compile()
    result=agent.invoke({
        "query":"简单的介绍下剑来里的宁瑶，不超过20个字",
        "messages":[],
        "final_answer":""
    })

    for msg in result["messages"]:
        if isinstance(msg,AIMessage):
            print(msg.content)


'''
[ceo]选这个部门research，原因是The request involves fact-finding and investigation, which aligns with the research department's focus on technical deep-dives.
[paper_review] 宁瑶是《剑来》中的重要角色，聪慧且坚韧，具备强大修炼潜力。
[web_search] 宁瑶是《剑来》中的女主角，聪慧坚韧，修炼天赋极高。

'''





if __name__ == "__main__":
    #test_single_partment()
    hierarchical_test()

    
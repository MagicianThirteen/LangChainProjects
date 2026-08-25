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

class QualityState(BaseModel):
    score:float= Field(description="Quality score from 0.0 to 1.0")
    feedback:str= Field(description="Specific feedback for improvement")
    approved:bool= Field(description="Whether the report meets quality standards")



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
        revision_note=""
        if state["iteration"]>0 and state.get('quality_feedback'):
            revision_note=f"\n\nIMPORTANT — This is revision #{state['iteration']}. "
            f"Address this feedback: {state['quality_feedback']}"
        
        report=creative_llm.invoke([
            SystemMessage(content="You are a report writer. Produce a well-structured research report "
                    "with these sections:\n"
                    "1. Executive Summary (2-3 sentences)\n"
                    "2. Key Findings (bullet points)\n"
                    "3. Analysis (1-2 paragraphs)\n"
                    "4. Recommendations (3 actionable items)\n\n"
                    "Use markdown formatting. Be specific and actionable."
                    f"{revision_note}"),
            HumanMessage(content=f"Topic: {state['topic']}\n\n"
                    f"Analysis:\n{state['analyst']}\n\n"
                    f"Raw findings:\n{json.dumps(state['findings'][:6], indent=2)}")
        ])

        return{
            "report":report.content,
            #这里可以写明，是草稿还是修改，不用再把report传进来
            "messages":[AIMessage(content=f"[report writer] {'revise' if state['iteration']>0 else 'draft'}")]
        }

    def quality_checker(state:ResearchState)->dict:
        check_llm=llm.with_structured_output(QualityState)
        response=check_llm.invoke([
            SystemMessage(content="You are a quality reviewer. Score the report on:\n"
                                "- Completeness: Does it cover the topic well?\n"
                                "- Clarity: Is it well-written and easy to understand?\n"
                                "- Actionability: Are recommendations specific?\n\n"
                                "Score from 0.0 to 1.0. Approve if score >= 0.7.\n"
                                "If this is iteration 2 or higher, be more lenient."),
            HumanMessage(content=f"Topic: {state['topic']}\n"
                                f"Iteration: {state['iteration']}\n\n"
                                f"Report:\n{state['report']}")                   
        ])
        approved=response.approved or state["iteration"]>=2
        return{
            'quality_score':response.score,
            'quality_feedback':response.feedback,
            'iteration':state['iteration']+1,
            'messages':[AIMessage(content=(
                                f"[QUALITY CHECK]: Score {response.score:.1f} — "
                                f"{'APPROVED' if approved else 'REVISION NEEDED'}: {response.feedback}"
                            ),
                            name="quality_checker")]

        }

    def quality_gate(state:ResearchState)->Literal["writer","end"]:
        if state["quality_score"]>=0.7 or state["iteration"]>=2:
            return "end"
        return "writer" 

    graph=StateGraph(ResearchState)
    graph.add_node("supervisor",supervisor)
    graph.add_node("research",research)
    graph.add_node("analyst",analyst)
    graph.add_node("writer",report_writer)
    graph.add_node("quality_checker",quality_checker)

    graph.add_edge(START,"supervisor")
    graph.add_conditional_edges("supervisor",dispatch_searches,["research"])
    graph.add_edge("research","analyst")
    graph.add_edge("analyst","writer")
    graph.add_edge("writer","quality_checker")
    graph.add_conditional_edges("quality_checker",quality_gate,{
        "writer":"writer",
        "end":END
    })

    agent=graph.compile()
    initial_state = {
            "messages": [],
            #这里限制字数没用，要每个节点都设置，老子的token啊啊啊啊啊啊
            "topic": "简单的说一下剑来里的宁瑶，不超过50个字",
            "search_queries": [],
            "findings": [],
            "analysis": "",
            "report": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "iteration": 0,
        }
    result=agent.invoke(initial_state)
    print(result['report'])

'''
# Research Report on Ning Yao in "Sword Comes"

## Executive Summary
Ning Yao is a pivotal female character in the novel "Sword Comes," showcasing her evolution from a young girl to a mature, resilient figure. Her complex personality and emotional depth significantly contribute to the narrative, reflecting broader themes of growth and the intricacies of human relationships.

## Key Findings
- Ning Yao evolves from innocence to strength, serving as a key supporting character.
- Her character embodies warmth and resilience, navigating challenges with courage.
- Interactions with the protagonist reveal complex emotional dynamics that contribute to her development.
- Despite her strengths, there is a lack of detailed exploration of the challenges she encounters.
- Ning Yao's journey represents multifaceted femininity within a male-dominated genre.

## Analysis
The character of Ning Yao is intricately woven into the narrative of "Sword Comes," serving as a representation of both personal growth and the broader themes of struggle and resilience. Her journey from a naive girl to a formidable presence illustrates the complexities of femininity, particularly in a genre that often sidelines female characters. While her strength and maturity are highlighted, the narrative could benefit from a deeper exploration of the specific hardships she faces, which would provide a more nuanced understanding of her character arc and its implications on the overall plot. This oversight leaves readers with a desire for more detailed context regarding the events that shape her transformation.

Ning Yao’s relationships, particularly with the protagonist, add layers of emotional depth to the story. The dynamics between them not only drive the plot but also showcase the intricate nature of human relationships, emphasizing the importance of emotional connections in character development. Thus, her role is not merely supportive; it is central to the narrative’s exploration of themes such as growth, resilience, and the complexities of human emotions.

## Recommendations
1. **Expand Character Backstory:** Incorporate additional flashbacks or narrative elements that detail the specific challenges Ning Yao faces, enhancing readers' understanding of her growth and resilience.
   
2. **Deepen Emotional Dynamics:** Develop more scenes that explore Ning Yao’s relationships with other characters, particularly the protagonist, to highlight the emotional intricacies and their impact on her character arc.

3. **Integrate Themes of Femininity:** Include discussions or reflections on femininity within the narrative, positioning Ning Yao as a representative figure in a male-dominated genre, thus enriching the thematic framework of the story.

'''

if __name__ == "__main__":
    multi_agent_research_system()

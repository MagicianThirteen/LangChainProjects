from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class SupervisorState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    next_agent:str
    task_finish:bool
    final_write:str

class RouteDicision(BaseModel):
    next:Literal["researcher", "writer", "critic", "FINISH"]=Field(description="he next agent to call, or FINISH if task is complete")
    reasoning:str=Field(description="Why this agent was chosen")

supervisor_llm=llm.with_structured_output(RouteDicision)

def supervisor_agent():
    def supervisor(state:SupervisorState)->dict:
        systemprompt='''
        You are a supervisor managing a team of specialists:

        1. researcher - Gathers information and facts
        2. writer - Creates content and text
        3. critic - Reviews and improves work

        Based on the conversation, decide which agent should act next.
        If the task is complete, respond with FINISH.

        Current conversation shows the progress so far.

        '''
        #messages=SystemMessage(content=systemprompt)+state["messages"]
        messages = [
            SystemMessage(content=systemprompt),
            *state["messages"]
]
        decision=supervisor_llm.invoke(messages)
        if decision.next=="FINISH":
            return{
                "next_agent":"FINISH",
                "task_finish":True
            }
        return{
            "messages":[AIMessage(content=f"[Supervisor] Routing to {decision.next}: {decision.reasoning}")],
            "next_agent":decision.next
        }

    def researcher(state:SupervisorState)->dict:
        task=next(m.content for m in state["messages"] if isinstance(m,HumanMessage)),""
        prompt=ChatPromptTemplate.from_messages(
            [("system","You are a research specialist. Gather facts and information relevant to the task. Be thorough but concise.",),
             ("human","Task context:\n{context}\n\nProvide your research findings.越短越好，不能超过50个词")]
        )
        response=llm.invoke(prompt.format_messages(context=task))
        return{
            "messages":[AIMessage(content=f"[Researcher] {response.content}")]
        }

    def write(state:SupervisorState)->dict:
        #这里只是取最近的五条参数
        prompt=ChatPromptTemplate.from_messages([
            ("system","You are a writing specialist. Create clear, engaging content based on the available information."),
            ("human","Previous work:\n{context}\n\nWrite the content.越短越好，不能超过20个词")
        ])
        context="\n".join([m.content for m in state["messages"][-5:]])
        response=llm.invoke(prompt.format_messages(context=context))
        return{
            "messages":[AIMessage(content=f"[Writer] {response.content}")]#这里最好标注是出自什么节点
        }

    def critic(state:SupervisorState)->dict:
        #根据最近的三条作为上下文依据
        context="\n".join([m.content for m in state["messages"][-3:]])
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system","You are a quality critic. Review the work and provide constructive feedback. If the work is good, say so."),
                ("human","Work to review:\n{context}\n\nProvide your critique.越短越好，不能超过20个词")
            ]
        )
        response=llm.invoke(prompt.format_messages(context=context))
        return{
            "messages":[AIMessage(content=f"[Critic] {response.content}")]
        }

    def finalize(state:SupervisorState)->dict:
        #倒着找最后写的那一版，还有如果没有找到要返回什么
        for m in reversed(state["messages"]):
            if isinstance(m,AIMessage) and "[Writer]" in m.content:
                return{
                    "final_write":m.content
                }
        return{
            "final_write":"task complete"
        }

    def route(state:SupervisorState)->Literal["researcher","writer","critic","FINISH"]:
        if state["task_finish"]:
            return "finalize"
        return state["next_agent"]

    graph=StateGraph(SupervisorState)
    
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", write)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor",route,{
                                                "researcher":"researcher",
                                                "writer":"writer",
                                                "critic":"critic",
                                                "finalize":"finalize"    })

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("finalize", END)
    agent=graph.compile()

    result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="介绍下《剑来》里的宁瑶,最多30个词以内"
                    )
                ],
                "next_agent": "",
                "task_finish": False,
                "final_write": "",
            }
        )
    
    print("Agent conversation:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")

    print(f"\n\nFinal Response:\n{result['final_write']}")

    '''
    Agent conversation:

[Supervisor] Routing to researcher: The writer needs more information about 宁瑶 from the source material to create an accurate and concise description....

[Researcher] 宁瑶是《剑来》中的重要角色，性格坚韧聪慧，出身名门，修炼天赋极高。她在故事中与主角关系密切，展现出强大的内心和独立精神。...

[Supervisor] Routing to writer: The researcher has provided the necessary information about 宁瑶, so now the writer can create a concise description based on that information....

[Writer] 宁瑶是《剑来》中的坚韧聪慧角色，出身名门，修炼天赋高，与主角关系密切。...

[Supervisor] Routing to critic: The writer has created a description, and now the critic should review and improve the content for clarity and conciseness....

[Critic] Clear and concise. Consider adding more context about 宁瑶's role in the story for depth....


Final Response:
[Writer] 宁瑶是《剑来》中的坚韧聪慧角色，出身名门，修炼天赋高，与主角关系密切。
    
    '''


if __name__ == "__main__":
    supervisor_agent()
    
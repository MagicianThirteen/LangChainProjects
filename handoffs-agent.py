
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class HandoffState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    current_agent:str
    handoff_reason:str
    context_summary:str

class HandoffDecision(BaseModel):
     handoff_to:Literal["sales","support","billing","end"]=Field(description="Which agent to hand off to")
     handoff_reason:str=Field(description="Reason for handoff")
     context=str=Field(description="Key context to pass to next agent")

triage_llm=llm.with_structured_output(HandoffDecision)


def handoff_agent():
    def triage(state:HandoffState)->dict:
         #身份
         system = """You are a customer service triage agent. Your job is to:
                 1. Understand the customer's need
                 2. Route to the appropriate specialist:
                    - sales: Product questions, purchases, upgrades
                    - support: Technical issues, bugs, how-to questions
                    - billing: Payments, invoices, refunds
                    - end: Simple questions you can answer directly
         
                 Analyze the customer's message and decide where to route them."""
         messages=[SystemMessage(content=system),*state["messages"]]
         decision=triage_llm.invoke(messages)
         if decision.handoff_to=="end":
              #再简短的回答下客户的问题
              messages=[
                                  SystemMessage(
                                      content="Provide a brief, helpful response to the customer."
                                  ),
                                  *state["messages"],
                              ]
              response=llm.invoke(messages)
              return{
                   "messages":[AIMessage(content=f"[Triage] {response.content}")],
                   "current_agent":"end",
                   
              }
         
         return{
              "messages":AIMessage(content=f"[triage] handoff to {decision.handoff_to} reason:{decision.handoff_reason}"),
              "current_agent":decision.handoff_to,
              "handoff_reason":decision.handoff_reason,
              "context_summary":decision.context
              
         }
    
    def sales(state:HandoffState)->dict:
        return{

        }
    
    def support(state:HandoffState)->dict:
            return{
                
            }

    def billing(state:HandoffState)->dict:
         return{
              
         }

    def handoffroute(state:HandoffState)->str:
         return

    graph=StateGraph(HandoffState)
    graph.add_node("triage",triage)
    graph.add_node("sales",sales)
    graph.add_node("support",support)
    graph.add_node("billing",billing)
    graph.add_edge(START,"triage")
    graph.add_conditional_edges("triage",handoffroute,{
         "sales":"sales",
         "support":"support",
         "billing":"billing"
    })
    graph.add_edge("sales",END)
    graph.add_edge("support",END)
    graph.add_edge("billing",END)
    agent=graph.compile()
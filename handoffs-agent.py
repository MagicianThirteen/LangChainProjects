
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
     context:str=Field(description="Key context to pass to next agent")

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
              "messages":[AIMessage(content=f"[triage] handoff to {decision.handoff_to} reason:{decision.handoff_reason}")],
              "current_agent":decision.handoff_to,
              "handoff_reason":decision.handoff_reason,
              "context_summary":decision.context
              
         }
    
    def sales(state:HandoffState)->dict:
        system = f"""You are a sales specialist. Context from triage: {state.get('context_summary', 'None')}
        
                    Help the customer with product questions and purchases.
                    Be helpful and informative, not pushy.最多用20个字以内"""
        response=llm.invoke([SystemMessage(content=system),*state["messages"]])
        
        return{
               "messages":AIMessage(content=f"[sales] {response.content}"),
               "current_agent":"sales_complete"
        }
    
    def support(state:HandoffState)->dict:
          system = f"""You are a technical support specialist. Context from triage: {state.get('context_summary', 'None')}
          
                  Help the customer with technical issues.
                  Be patient and provide step-by-step guidance.最多用20个字以内"""
          
          response = llm.invoke([SystemMessage(content=system), *state["messages"]])

          return {
               "messages": [AIMessage(content=f"[Support] {response.content}")],
               "current_agent": "support_complete",
          }

    def billing(state:HandoffState)->dict:
          system = f"""You are a billing specialist. Context from triage: {state.get('context_summary', 'None')}
          
                    Help the customer with billing questions.
                    Be clear about policies and next steps.最多用20个字以内"""

          response = llm.invoke([SystemMessage(content=system), *state["messages"]])

          return {
          "messages": [AIMessage(content=f"[Billing] {response.content}")],
          "current_agent": "billing_complete",
          }

    def handoffroute(state:HandoffState)->str:
          agent = state["current_agent"]
          if agent in ["sales", "support", "billing"]:
               return agent
          return "end"
    
    graph=StateGraph(HandoffState)
    graph.add_node("triage",triage)
    graph.add_node("sales",sales)
    graph.add_node("support",support)
    graph.add_node("billing",billing)
    graph.add_edge(START,"triage")
    graph.add_conditional_edges("triage",handoffroute,{
         "sales":"sales",
         "support":"support",
         "billing":"billing",
         "end":END
    })
    graph.add_edge("sales",END)
    graph.add_edge("support",END)
    graph.add_edge("billing",END)
    agent=graph.compile()
    
    queries = [
            "My app keeps crashing when I try to upload photos",
            "I want to upgrade to the premium plan",
            "I was charged twice for my subscription",
            "What time do you close?",
        ]
    for query in queries:
            print(f"Customer: {query}")
    
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=query+"最多用20个字以内解释")],
                    "current_agent": "",
                    "handoff_reason": "",
                    "context_summary": "",
                }
            )
    
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"  {msg.content[:150]}...")
    
            print("-" * 50)

'''
Customer: My app keeps crashing when I try to upload photos
  [triage] handoff to support reason:Technical issue with the app....
  [Support] 请检查网络连接并重启应用。...
--------------------------------------------------
Customer: I want to upgrade to the premium plan
  [triage] handoff to sales reason:Customer is requesting an upgrade....
  [sales] Great choice! The premium plan offers enhanced features. Let me assist you with the upgrade process....
--------------------------------------------------
Customer: I was charged twice for my subscription
  [triage] handoff to billing reason:Customer is inquiring about a billing issue....
  [Billing] 请提供您的账户信息，我们将调查重复收费问题。...
--------------------------------------------------
Customer: What time do you close?
  [Triage] 我们的营业时间是晚上9点。...
--------------------------------------------------
'''
    
    
if __name__ == "__main__":
    handoff_agent()
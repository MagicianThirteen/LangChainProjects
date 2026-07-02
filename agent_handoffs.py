
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

llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

class HandoffState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    current_agent:str
    handoff_reason:str
    context_summary:str

class HandOffDecision(BaseModel):
    handoff_to:Literal["sales","support","billing","stay","end"]=Field(
        description="Which agent to hand off to"

    )
    reason:str=Field(description="Reason for handoff")
    #需要简单的告知前面发生了什么
    context:str=Field(description="Key context to pass to next agent")

def create_customer_service_system():   
#triage_agent:负责分发agent
    def triage_agent(state:HandoffState)->dict:
        #system:职责描述
        system='''
        You are a customer service triage agent. Your job is to:
            1. Understand the customer's need
            2. Route to the appropriate specialist:
            - sales: Product questions, purchases, upgrades
            - support: Technical issues, bugs, how-to questions
            - billing: Payments, invoices, refunds
            - end: Simple questions you can answer directly

            Analyze the customer's message and decide where to route them.

        '''
        #和用户问题，拼接成list[basemesages]
        messages=[SystemMessage(content=system)]+state["messages"]
        #为了更准确的知道去往哪个节点，llm用结构化输出
        handoff_llm=llm.with_structured_output(HandOffDecision)
        decision=handoff_llm.invoke(messages)
        #根据输出结果，决定去往哪里，以及数据更新
        if decision.handoff_to=="end":
            messages=[SystemMessage(content="Provide a brief, helpful response to the customer."),
                    *state["messages"],]
            response=llm.invoke(messages)
            return{
                "messages":[AIMessage(content=f"[Triage]:{response.content}")],
                "current_agent":"end",

            }
        return{
            "messages":[AIMessage(content=f"[Triage] Transferring to {decision.handoff_to}: {decision.reason}")],
            "current_agent":decision.handoff_to,
            "handoff_reason":decision.reason,
            "context_summary":decision.context
        }

    def sales_agent(state:HandoffState)->dict:
        system=f"""
        You are a sales specialist. Context from triage: {state.get('context_summary', 'None')}

                Help the customer with product questions and purchases.
                Be helpful and informative, not pushy.

        """
        messages=[
            SystemMessage(content=system),
            *state["messages"]
        ]
        response=llm.invoke(messages)
        return{
            "messages":[AIMessage(content=f"[sales]:{response.content}")],
            "current_agent":"sales_complete",
        }

    def support_agent(state: HandoffState) -> dict:
            """Technical support specialist."""
            system = f"""You are a technical support specialist. Context from triage: {state.get('context_summary', 'None')}

            Help the customer with technical issues.
            Be patient and provide step-by-step guidance."""

            response = llm.invoke([SystemMessage(content=system), *state["messages"]])

            return {
                "messages": [AIMessage(content=f"[Support] {response.content}")],
                "current_agent": "support_complete",
            }

    def billing_agent(state: HandoffState) -> dict:
            """Billing specialist."""
            system = f"""You are a billing specialist. Context from triage: {state.get('context_summary', 'None')}

            Help the customer with billing questions.
            Be clear about policies and next steps."""

            response = llm.invoke([SystemMessage(content=system), *state["messages"]])

            return {
                "messages": [AIMessage(content=f"[Billing] {response.content}")],
                "current_agent": "billing_complete",
            }

    def route_from_triage(state:HandoffState)->str:
     agent=state["current_agent"]
     if agent in ["sales","support","billing"]:
          return agent
     return "end"
        
    
    graph=StateGraph(HandoffState)
    graph.add_node("triage", triage_agent)
    graph.add_node("sales", sales_agent)
    graph.add_node("support", support_agent)
    graph.add_node("billing", billing_agent)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {"sales": "sales", "support": "support", "billing": "billing", "end": END}
    )

    graph.add_edge("sales", END)
    graph.add_edge("support", END)
    graph.add_edge("billing", END)

    return graph.compile()


def demo_handoffs():
     agent=create_customer_service_system()
     queries=[
         "My app keeps crashing when I try to upload photos",
        "I want to upgrade to the premium plan",
        "I was charged twice for my subscription",
        "What time do you close?", 
     ]
     for query in queries:
        print(f"Customer: {query}")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=query)],
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
  [Triage] Transferring to support: The issue involves a technical problem with the app....
  [Support] I'm sorry to hear that you're experiencing crashes while uploading photos. Let's work through this step-by-step to identify and resolve the ...
--------------------------------------------------
Customer: I want to upgrade to the premium plan
  [Triage] Transferring to sales: Customer is inquiring about an upgrade, which falls under product questions....
  [sales]:Great to hear that you're interested in upgrading to the premium plan! The premium plan offers several enhanced features that can really enhan...
--------------------------------------------------
Customer: I was charged twice for my subscription
  [Triage] Transferring to billing: Customer is inquiring about a payment issue related to their subscription....
  [Billing] I’m sorry to hear that you were charged twice for your subscription. I can help you with that.

First, could you please provide me with the ...
--------------------------------------------------
Customer: What time do you close?
  [Triage]:Our closing time varies by location. Please check our website or contact your local store for specific hours....
--------------------------------------------------
'''


if __name__ == "__main__":
    demo_handoffs()

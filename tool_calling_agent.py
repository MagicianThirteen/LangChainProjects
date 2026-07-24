from urllib import response
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
import operator
import json
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: calculate('2 + 2')"""
    try:
        result = eval(expression)  # Note: In production, use a safe math parser
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Clear",
        "paris": "65°F, Partly Cloudy",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"

class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def create_tool_agent():
    tools=[calculate,get_weather]
    llm_with_tools=llm.bind_tools(tools)

    def agent_node(state:AgentState)->dict:
        response=llm_with_tools.invoke(state["messages"])
        return {"messages":[response]}

    def should_continue(state:AgentState)->Literal["tools","end"]:
        last_message=state['messages'][-1]

        if not hasattr(last_message,"tool_calls") or not last_message.tool_calls:
            return "end"
        return "tools"

    tool_node=ToolNode(tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent") 

    graph.add_conditional_edges("agent",should_continue,{
        "tools":"tools",
        "end":END
    })  

    graph.add_edge("tools", "agent")
    agent=graph.compile() 
    return agent

def tool_agent():
    agent=create_tool_agent()
    queries = [
            "What's 25 * 17?",
            "What's the weather in Tokyo?",
            "What's 100 / 4 and what's the weather in London?",
        ]

    for query in queries:
        print(f"Query: {query}")
        
        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}")
        print(f"Total messages: {len(result['messages'])}")
        print("-" * 40)


    '''
    Query: What's 25 * 17?
Response: The result of \( 25 \times 17 \) is 425.
Total messages: 4
----------------------------------------
Query: What's the weather in Tokyo?
Response: The weather in Tokyo is currently 68°F and clear.
Total messages: 4
----------------------------------------
Query: What's 100 / 4 and what's the weather in London?
Response: The result of \( 100 / 4 \) is 25.0. 

As for the weather in London, it is currently 58°F and cloudy.
Total messages: 5
----------------------------------------
    '''



if __name__ == "__main__":
    tool_agent()       
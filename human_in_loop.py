from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def interrupt_for_approval():
    #根据我一个主题，生成一小段话，然后我审批这个话行不行
    #行就完成，不行就再生成一次
    class approvalstate(TypedDict):
        topic:str
        draft:str
        approval:bool
        feedback:str
        final:str
    
    def create_draft(state:approvalstate)->dict:
        prompt=f"请根据这个话题{state['topic']},生成一个20字左右的说明"
        content=llm.invoke(prompt)
        return {
            "draft":content.content
        }
    
    def wait_approval(state:approvalstate)->dict:
        return state
    
    def finalize(state:approvalstate)->dict:
        if state["approval"]==True:
            return{
                "feedback":state["feedback"],
                "final":state["draft"]
            }
        else:
            prompt=f"请根据反馈{state['feedback']},修改一下关于这个主题{state['topic']}的这个回答{state['draft']},字数50字以内"
            content=llm.invoke(prompt)
            return{
                "feedback":state["feedback"],
                "final":content.content
            }
    
    graph=StateGraph(approvalstate)
    graph.add_node("draft",create_draft)
    graph.add_node("approval",wait_approval)
    graph.add_node("final",finalize)

    graph.add_edge(START,"draft")
    graph.add_edge("draft","approval")
    graph.add_edge("approval","final")
    graph.add_edge("final",END)

    memorysave=MemorySaver()
    agent=graph.compile(
        checkpointer=memorysave,
        interrupt_before=["approval"]
    )
    config={"configurable":{"thread_id":"demo_1"}}

    draft=agent.invoke({
        "topic":"剑来宁瑶",
        "approval":False,
        "draft":"",
        "feedback":"",
        "final":""
    },config)

    print(f"第一阶段得到的草稿是{draft['draft']}")
    print("当前的状态是")
    current_state=agent.get_state(config)
    print(f"next node:{current_state.next}")
    print(f"State keys: {list(current_state.values.keys())}")
    print("开始人工审批")
    agent.update_state(config,{
        "approval":False,
        "feedback":"换一种更简练的说法"
    })
    result=agent.invoke(None,config)
    print(f"之前得到的草稿是{result['draft']}")
    print(f"最终得到的文稿是{result['final']}")

'''
第一阶段得到的草稿是宁瑶是《剑来》中的重要角色，聪慧坚定，深具武学天赋，伴随主角成长与冒险。
当前的状态是
next node:('approval',)
State keys: ['topic', 'draft', 'approval', 'feedback', 'final']
开始人工审批
之前得到的草稿是宁瑶是《剑来》中的重要角色，聪慧坚定，深具武学天赋，伴随主角成长与冒险。
最终得到的文稿是宁瑶是《剑来》中的重要角色，聪慧坚定，武学天赋出众，伴随主角共同成长与冒险。
'''


if __name__ == "__main__":
    interrupt_for_approval()

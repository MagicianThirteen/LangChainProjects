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

#循环完善宁瑶介绍的
def iterative_review():
    class ReviewState(TypedDict):
        document:str
        review_comments:list[str]
        status:str
        revision_count:int
        final_doc:str
    
    def submit(state:ReviewState)->dict:
        print("submit node")
        print(f"准备审核的文本{state['document']}")
        return state
    
    def wait_apply(state:ReviewState)->dict:
        if state["review_comments"]==[]:
           return state
        feedback=state["review_comments"][-1]
        prompt=f"根据返回的信息{feedback}修改文档{state['document']}"
        content=llm.invoke(prompt)
        return{
            "document":content.content,
            "revision_count":state["revision_count"]+1
        }
    
    def route(state:ReviewState)->Literal["apply","done"]:
        if state["status"]=="approve":
           return "done"
        else:
           return "apply"
    
    def done(state:ReviewState)->dict:
        return{
            "final_doc":state["document"]
        }
    
    graph=StateGraph(ReviewState)
    graph.add_node("submit",submit)
    graph.add_node("apply",wait_apply)
    graph.add_node("done",done)
    graph.add_edge(START,"submit")
    graph.add_conditional_edges("submit",route,
                                {"apply":"apply",
                                 "done":"done"})
    graph.add_edge("apply","submit")
    graph.add_edge("done",END)

    agent=graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["submit"]
    )

    config = {"configurable": {"thread_id": "demo-2"}}

    #第一介绍宁瑶
    result=agent.invoke({
        "document":"宁瑶",
        "final_doc":"",
        "review_comments":[],
        "revision_count":0,
        "status":""
    },config)
    print(f"第一次介绍宁瑶{result['document']}")
    feedback1="简短的介绍下剑来的宁瑶，20个字以内"
    agent.update_state(config,{
        "review_comments":[feedback1],
        "status":"need review"
    },)

    result=agent.invoke(None,config)
    print(f"第二次介绍宁瑶{result['document']}")

    feedback2="再多点介绍"
    agent.update_state(config,{
        "review_comments":[feedback2],
        "status":"need review"
    },)
    result=agent.invoke(None,config)
    print(f"第三次介绍宁瑶{result['document']}")
    agent.update_state(config,{
        "status":"approve"
    },)
    result=agent.invoke(None,config)
    print(f"最终介绍宁瑶{result['final_doc']}")

'''
submit node
准备审核的文本宁瑶
第二次介绍宁瑶宁瑶是剑来中的重要角色，聪慧坚定，勇敢追求理想。
submit node
准备审核的文本宁瑶是剑来中的重要角色，聪慧坚定，勇敢追求理想。
第三次介绍宁瑶宁瑶是《剑来》中的重要角色之一，她以聪慧和坚定的性格深受读者喜爱。作为一个勇敢追求理想的女性角色，宁瑶在故事中展现了非凡的智慧和勇气。她不仅在面对困难和挑战时毫不退缩，还始终坚持自己的信念，努力实现自己的目标。

在《剑来》的世界中，宁瑶的成长历程充满了波折与挑战。她经历了许多磨难，但始终保持着对理想的执着追求。她的聪慧使她能够在复杂的局势中做出明智的决策，而她的坚定则让她在逆境中不屈不挠，勇往直前。

宁瑶的角色不仅仅是一个追求理想的勇者，她还展现了深厚的人性关怀和对朋友的忠诚。在与其他角色的互动中，她的智慧和勇气常常能够激励身边的人，成为他们的精神支柱。

总的来说，宁瑶是一个多层次的角色，她的聪慧、坚定和勇敢使她在《剑来》的故事中扮演了不可或缺的角色，深刻影响着整个故事的发展。
submit node
准备审核的文本宁瑶是《剑来》中的重要角色之一，她以聪慧和坚定的性格深受读者喜爱。作为一个勇敢追求理想的女性角色，宁瑶在故事中展现了非凡的智慧和勇气。她不仅在面对困难和挑战时毫不退缩，还始终坚持自己的信念，努力实现自己的目标。

在《剑来》的世界中，宁瑶的成长历程充满了波折与挑战。她经历了许多磨难，但始终保持着对理想的执着追求。她的聪慧使她能够在复杂的局势中做出明智的决策，而她的坚定则让她在逆境中不屈不挠，勇往直前。

宁瑶的角色不仅仅是一个追求理想的勇者，她还展现了深厚的人性关怀和对朋友的忠诚。在与其他角色的互动中，她的智慧和勇气常常能够激励身边的人，成为他们的精神支柱。

总的来说，宁瑶是一个多层次的角色，她的聪慧、坚定和勇敢使她在《剑来》的故事中扮演了不可或缺的角色，深刻影响着整个故事的发展。
最终介绍宁瑶宁瑶是《剑来》中的重要角色之一，她以聪慧和坚定的性格深受读者喜爱。作为一个勇敢追求理想的女性角色，宁瑶在故事中展现了非凡的智慧和勇气。她不仅在面对困难和挑战时毫不退缩，还始终坚持自己的信念，努力实现自己的目标。

在《剑来》的世界中，宁瑶的成长历程充满了波折与挑战。她经历了许多磨难，但始终保持着对理想的执着追求。她的聪慧使她能够在复杂的局势中做出明智的决策，而她的坚定则让她在逆境中不屈不挠，勇往直前。

宁瑶的角色不仅仅是一个追求理想的勇者，她还展现了深厚的人性关怀和对朋友的忠诚。在与其他角色的互动中，她的智慧和勇气常常能够激励身边的人，成为他们的精神支柱。

总的来说，宁瑶是一个多层次的角色，她的聪慧、坚定和勇敢使她在《剑来》的故事中扮演了不可或缺的角色，深刻影响着整个故事的发展。

'''






if __name__ == "__main__":
    #interrupt_for_approval()
    iterative_review()

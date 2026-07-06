from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import operator
from dotenv import load_dotenv

load_dotenv()

llm=init_chat_model("gpt-4o-mini", temperature=0.0)

class CodeGenState(TypedDict):
    task:str
    code:str
    errors:Annotated[list[str],operator.add]
    iteration:int
    max_iter:int
    success:bool

def self_correcting_code():

    def generate_code(state:CodeGenState)->dict:
        #第一次是直接生成，第二次是直接修改
        if state["iteration"]==0:
            prompt=f"Write Python code for: {state['task']}\nReturn only the code."
        else:
            prompt = (
                f"Fix this Python code:\n{state['code']}\n\n"
                #根据最新返回的错误
                f"Errors:\n{state['errors'][-1]}\n\n"
                "Return only the corrected code."
            )
        
        response=llm.invoke(prompt)
        code=response.content.strip()

        #防御性编程，如果碰上了marddown
        if code.startswith("```"):
            code=code.split("```")[1]
            if code.startswith("python"):
                code=code[6:]

        return{
            "code":code,
            "iteration":state["iteration"]+1
        }
    
    def validate_code(state:CodeGenState)->dict:
        #编译错误，运行错误，功能错误，边界错误
        code=state["code"]
        try:
            compile(code,"<string>","exec")
        except SyntaxError as e:
            return {
                "errors":[f"SyntaxError: {e}"],
                "success":False
            }
        
        test_cases = [
            ([3, 1, 4, 1, 5, 9], 5),  # normal case
            ([1, 1, 1], None),  # all same → no second largest
            ([7], None),  # single element
            ([3, -1, 3, 5, 5], 3),  # duplicates at top
        ]

        namespace={}
        try:
            exec(code,namespace)
        except Exception as e:
            return{
                "errors":[f"Runtime error:{e}"],
                "success":False
            }
        
        if "solve" not in namespace:
            return{"errors":["Function 'solve' not found in code"],
                   "success":False}
    
        for inputs,expected in test_cases:
            try:
                result=namespace["solve"](inputs)
                if result!=expected:
                    return {
                        "errors": [
                            f"solve({inputs}) returned {result}, expected {expected}"
                        ],
                        "success": False,
                    }
            except Exception as e:
                return {
                    "errors":[f"solve({inputs}) raised {e}"],
                    "success":False
                }
        return {"success":True}
    
    def should_continue(state:CodeGenState)->Literal["generate","end"]:
        if state["success"]:
            return "end"
        elif state["iteration"] >= state["max_iter"]:
            return "end"
        else:
            return "generate"
    def finalize(state: CodeGenState) -> dict:
        return state
    
    graph = StateGraph(CodeGenState)

    graph.add_node("generate", generate_code)
    graph.add_node("validate", validate_code)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate", should_continue, {"generate": "generate", "end": "finalize"}
    )  # Loop back to "generate" if not successful and under max iterations, otherwise go to "finalize"
    graph.add_edge("finalize", END)

    app = graph.compile()

    # visualize the graph
    print("\n--- Mermaid Graph ---")
    # print(app.get_graph().draw_mermaid())

    # save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_code.png", "wb") as f:
        f.write(png_bytes)
    print("\nGraph saved to graph_code.png")

    print("Self-Correcting Code Generator:\n")

    result = app.invoke(
        {
            "task": "a function that calculates factorial recursively",
            "code": "",
            "errors": [],
            "iteration": 0,
            "max_iter": 3,
            "success": False,
        }
    )

    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"Final Code:\n{result['code']}")

    '''
    Task: a function that calculates factorial recursively
Iterations: 3
Success: False
Final Code:

def solve(n):
    def factorial(n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * factorial(n - 1)
    
    return [factorial(x) for x in n]
    '''


if __name__ == "__main__":
    self_correcting_code()



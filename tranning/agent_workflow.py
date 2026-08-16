# Install: pip install langgraph langchain-core
import json, re
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

# 01. define the Agent Status
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 02. define claculate tool
def execute_calc(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"

# 03. define Node Logic
def agent_node(state: AgentState):
    """Agent 決策節點 (實際部屬時替換為你訓練好的本地模型 API)"""
    messages = state["messages"]
    last_msg = messages[-1].content

    # 模擬微調模型輸出的 ReAct 軌跡
    if isinstance(messages[-1], ToolMessage):
        return {"messages": [AIMessage(content=f"Final Answer: 計算結果為 {last_msg}")]}

    # 發起工具呼叫
    action_payload = json.dumps({"action": "calc", "input": "(35 + 45) * 12"})
    return {"messages": [AIMessage(content=f"Thought: 需要執行計算\nAction: ```json\n{action_payload}\n```")]}

def tool_node(state: AgentState):
    """工具執行節點"""
    last_msg = state["messages"][-1].content
    match = re.search(r"```json\s*(\{.*?\})", last_msg, re.DOTALL)
    if match:
        data = json.loads(match.group(1))
        result = execute_calc(data["input"])
        return {"messages": [ToolMessage(content=result, tool_call_id = "call_01")]}
    return {"messages": [ToolMessage(content="Tool Exection Failed", tool_call_id="call_01")]}
# 04. 判斷路由條件(Conditional Edge)
def should_continue(state: AgentState):
    last_msg = state["messages"][-1].content
    return "tools" if "Action:" in last_msg else END

# 05. 建構工作劉塗(Workflow Graph)
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 06. run and test
inputs = {"messages": [SystemMessage(content="You are a helpful agent."), HumanMessage(content="幫我算(35 + 45) * 12")]}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"--- 節點: {key} ---")
        print(value["messages"][-1].content)
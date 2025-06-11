from typing import TypedDict,Annotated,Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

load_dotenv()

class Agentstate(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int , b:int):
    """This is an addition function that adds two number"""
    return a + b

def sub(a:int , b:int):
    """This is a subtraction function that subtracts two number"""
    return a - b

def mul(a:int , b:int):
    """This is a multiplication function that multiplies two number"""
    return a * b

tools = [add,sub,mul]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001").bind_tools(tools)

def model_call(state:Agentstate)-> Agentstate:
    SYSTEM_PROMPT=SystemMessage(content="You are my AI agent, please answer my query to the best of your ability")
    response = model.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages":[response]}

def should_cont(state:Agentstate)->Agentstate:
    messages = state["messages"]
    last_messsage = messages[-1]
    if not last_messsage.tool_calls:
        return "end"
    else:
        return "continue"
    
graph=StateGraph(Agentstate)
graph.add_node("agent",model_call)
tool_node=ToolNode(tools=tools)
graph.add_node("tool",tool_node)
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_cont,
    {
        "continue":"tool",
        "end":END,
    },
)

graph.add_edge("tool","agent")

agent= graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print() 

inputs = {"messages":[("user","Add 6 and 7 and multiply the result with 3 and then subtract -1 from it")]}
print_stream(agent.stream(inputs,stream_mode="values"))
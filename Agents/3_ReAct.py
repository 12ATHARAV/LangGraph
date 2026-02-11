from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()


# # Annotated - provides additional context without affecting the type itself
# email = Annotated[str, "This has to be a valid email format!"]
# print(email.__metadata__)


# Sequence -- To automatically handle the state updates for sequences such as by adding new messages to a chat history


# Reducer Function
# Rule that controls how updates from nodes are combined with the existing state.
# Tells us how to merge new data into the current state

# Without a reducer, updates would have replaced the existing value entirely!

# # Without a reducer
# state = {"messages": ["Hi!"]}
# update = {"messages": ["Nice to meet you!"]}
# new_state = {"messages": ["Nice to meet you!"]}

# # With a reducer
# state = {"messages": ["Hi!"]}
# update = {"messages": ["Nice to meet you!"]}
# new_state = {"messages": ["Hi!", "Nice to meet you!"]}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numer together"""
    return a+b
    
tools = [add]

model = ChatGroq(model = "qwen/qwen3-32b").bind_tools(tools)
    
def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "you are my AI assistant, please answer my query to best of your ability"                              
    )
    response = model.invoke([system_prompt])
    return {"messages": [response]}
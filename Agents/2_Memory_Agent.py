from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


from dotenv import load_dotenv
load_dotenv()


class AgentState(TypedDict):
    # messages: List[HumanMessage]
    # messages_ai: List[AIMessage]
    messages: List[Union[HumanMessage, AIMessage]]
    
    
llm = ChatGroq(model = "qwen/qwen3-32b")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    # print(result["messages"])
    conversation_history = result["messages"]
    
    user_input = input("Enter: ")

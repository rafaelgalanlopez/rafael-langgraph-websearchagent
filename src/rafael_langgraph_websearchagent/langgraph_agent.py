import os
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# Cargar variables de entorno si existe un archivo .env
load_dotenv()

# 1. Configuración del modelo de lenguaje
model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
)

# 2. Definición de herramientas (Tools)
search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

tools = [multiply, search_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Vinculación de herramientas al modelo
model_with_tools = model.bind_tools(tools)


# 3. Definición del estado del agente
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# 4. Definición de los Nodos
def llm_call(state: dict):
    """Nodo que invoca al modelo de lenguaje."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Nodo que ejecuta las herramientas solicitadas por el modelo."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# 5. Definición de la lógica condicional
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide si se continua a la ejecución de herramientas o se finaliza."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END


# 6. Construcción y compilación del flujo de trabajo
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Agente compilado listo para usar/importar
agent = agent_builder.compile()
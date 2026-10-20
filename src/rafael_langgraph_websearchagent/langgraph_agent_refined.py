import os
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

#CARGAR VARIABLES DE ENTORNO
load_dotenv()

#MODELO DE LENGUAJE
model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
)

#HERRAMIENTA WEB SEARCH
search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}

#VINCULAR LISTA DE HERRAMIENTAS AL MODELO
model_with_tools = model.bind_tools(tools)

#ESTADO DEL AGENTE (State)
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    tool_tokens: int  # Tokens consumidos en llamadas donde el modelo decidió usar tools

#FUNCIONES AUXILIARES AL MODELO
#Inspección de ToolMessages por consola
def print_tool_message(tool_name: str, args: dict, content: str):
    """Imprime el contenido de un ToolMessage de forma clara y legible por consola."""
    print("\n" + "=" * 60)
    print(f"🛠️  [TOOL EXECUTION] -> {tool_name}")
    print(f"📥 Arguments: {args}")
    print("-" * 60)
    print(f"📄 Output:\n{content}")
    print("=" * 60 + "\n")

#Cálculo y seguimiento del uso de tokens
def calculate_token_usage(response, current_state: dict) -> dict:
    """Extrae los tokens de la respuesta del LLM y calcula los acumulados para el estado."""
    token_usage = response.usage_metadata or {}
    p_tokens = token_usage.get("input_tokens", 0)
    c_tokens = token_usage.get("output_tokens", 0)
    t_tokens = token_usage.get("total_tokens", p_tokens + c_tokens)

    is_tool_call = bool(response.tool_calls)
    tool_tokens_added = t_tokens if is_tool_call else 0

    return {
        "total_tokens": current_state.get("total_tokens", 0) + t_tokens,
        "prompt_tokens": current_state.get("prompt_tokens", 0) + p_tokens,
        "completion_tokens": current_state.get("completion_tokens", 0) + c_tokens,
        "tool_tokens": current_state.get("tool_tokens", 0) + tool_tokens_added,
    }

#NODOS DEL GRAFO
def llm_call(state: dict):
    """Nodo que invoca al modelo de lenguaje y rastrea el uso de tokens."""
    system_prompt = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Your static knowledge base has a cutoff of July 2025. "
            "We are in October 2026."
            "For any information or events prior to July 2025, rely on your static knowledge. "
            "For real-time, current information, or facts after July 2025 that you do not know, "
            "you MUST use the web search tool provided."
        )
    )

    response = model_with_tools.invoke([system_prompt] + state["messages"])

    # Extracción de métricas de tokens usando la función auxiliar
    token_metrics = calculate_token_usage(response, state)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        **token_metrics,
    }

def tool_node(state: dict):
    """Nodo que ejecuta las herramientas solicitadas por el modelo e inspecciona el ToolMessage."""
    result = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        # Formatear el contenido si devuelve una estructura no-string
        content_str = str(observation)

        # Inspección / Impresión fácil de leer del ToolMessage
        print_tool_message(
            tool_name=tool_call["name"],
            args=tool_call["args"],
            content=content_str
        )

        result.append(
            ToolMessage(
                content=content_str,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": result}

#ENRUTAMIENTO DE NODOS
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide si se continúa a la ejecución de herramientas o se finaliza."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END

#CONSTRUCCIÓN Y COMPILAMIENTO DEL GRAFO
agent_builder = StateGraph(MessagesState)

#INTEGRACIÓN DE NODOS
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

#INTEGRACIÓN DE NODOS Y SUS ENLACES
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

#AGENTE COMPILADO
agent = agent_builder.compile()
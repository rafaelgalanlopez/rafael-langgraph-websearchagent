#1 IMPORTS Y DEPENDENCIAS
import os
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END

#2 VARIABLES DE ENTORNO
load_dotenv()

#3 INSTANCIAR EL MODELO
model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
)

#4 DECLARAR HERRAMIENTAS (Tavily Search for Internet)
search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

#5 DECLARAR ESTRUCTURA DE DATOS Y DEL MODELO
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    tool_tokens: int
    used_tools: Annotated[list[str], operator.add]

#6 HERRAMIENTA AUXILIAR: IMPRIMIR HERRAMIENTAS USADAS EN EL PROMPT
def print_tool_message(tool_name: str, args: dict, content: str):
    print("\n" + "=" * 60)
    print(f"🛠️  [TOOL EXECUTION] -> {tool_name}")
    print(f"📥 Arguments: {args}")
    print("-" * 60)
    print(f"📄 Output:\n{content}")
    print("=" * 60 + "\n")

#7 HERRAMIENTA AUXILIAR: CALCULAR USO DE TOKENS DE LA RESPUESTA DEL LLM
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

#7.1 HERRAMIENTA AUXILIAR: IMPRIMIR REPORTE DE TOKENS Y HERRAMIENTAS
def print_token_and_tool_report(state: dict):
    """Imprime por consola el informe detallado de herramientas y métricas de tokens acumulados."""
    tools_used = state.get("used_tools", [])
    unique_tools = list(set(tools_used))

    tools_fmt = (
        "\n".join([f"  • {t} (usada {tools_used.count(t)} vez/veces)" for t in unique_tools])
        if unique_tools
        else "  • Ninguna (respuesta basada en conocimiento estático)"
    )

    print("\n" + "=" * 40)
    print("🧰 HERRAMIENTAS UTILIZADAS")
    print("=" * 40)
    print(f"{tools_fmt}")
    print("\n" + "=" * 40)
    print("📊 MÉTRICAS DE USO DE TOKENS ACUMULADOS")
    print("=" * 40)
    print(f"🔹 Llamadas al LLM    : {state.get('llm_calls', 0)}")
    print(f"🔹 Tokens de Prompt    : {state.get('prompt_tokens', 0)}")
    print(f"🔹 Tokens de Respuesta : {state.get('completion_tokens', 0)}")
    print(f"🔹 Tokens Totales      : {state.get('total_tokens', 0)}")
    print(f"🛠️ Tokens en Tool Calls: {state.get('tool_tokens', 0)}")
    print("=" * 40 + "\n")

#8 NODO PRINCIPAL 
def llm_call(state: dict):
    system_prompt = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Your static knowledge base has a cutoff of July 2025. "
            "We are in October 2026. "
            "For any information or events prior to July 2025, rely on your static knowledge. "
            "For real-time, current information, or facts after July 2025 that you do not know, "
            "you MUST use the web search tool provided."
        )
    )

    response = model_with_tools.invoke([system_prompt] + state["messages"])
    token_metrics = calculate_token_usage(response, state)
    tools_called = [tc["name"] for tc in response.tool_calls] if response.tool_calls else []

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "used_tools": tools_called,
        **token_metrics,
    }

#9 NODO DE HERRAMIENTAS
def tool_node(state: dict):
    result = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        content_str = str(observation)

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

#10 NODO FINAL: Retorna el consolidado de tokens y herramientas como un AIMessage al final del flujo
def report_node(state: dict):
    tools_used = state.get("used_tools", [])
    unique_tools = list(set(tools_used))
    
    tools_fmt = (
        "\n".join([f"  • {t} (usada {tools_used.count(t)} vez/veces)" for t in unique_tools])
        if unique_tools
        else "  • Ninguna (respuesta basada en conocimiento estático)"
    )

    report_text = (
        "\n" + "=" * 40 + "\n"
        "🧰 HERRAMIENTAS UTILIZADAS\n"
        + "=" * 40 + "\n"
        f"{tools_fmt}\n"
        "\n" + "=" * 40 + "\n"
        "📊 MÉTRICAS DE USO DE TOKENS ACUMULADOS\n"
        + "=" * 40 + "\n"
        f"🔹 Llamadas al LLM    : {state.get('llm_calls', 0)}\n"
        f"🔹 Tokens de Prompt    : {state.get('prompt_tokens', 0)}\n"
        f"🔹 Tokens de Respuesta : {state.get('completion_tokens', 0)}\n"
        f"🔹 Tokens Totales      : {state.get('total_tokens', 0)}\n"
        f"🛠️ Tokens en Tool Calls: {state.get('tool_tokens', 0)}\n"
        + "=" * 40
    )

    return {"messages": [AIMessage(content=report_text)]}

#11 CONDICIONAL Y CONSTRUCCIÓN DEL GRAFO
def should_continue(state: MessagesState) -> Literal["tool_node", "report_node"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return "report_node"

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("report_node", report_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "report_node"])
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("report_node", END)

agent = agent_builder.compile()
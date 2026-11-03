#1 IMPORTS Y DEPENDENCIAS
import os
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch, TavilyExtract
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END

#2 VARIABLES DE ENTORNO
load_dotenv()

#3 INSTANCIAR EL MODELO
model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
)

#3.1 EVALUADOR DE RELEVANCIA CON PYDANTIC SCHEMA (FLEXIBILIZADO)
class RelevanceGrade(BaseModel):
    es_relevante: bool = Field(
        description=(
            "Indica 'True' si los resultados obtenidos contienen información útil o responden AL MENOS A UN ASPECTO "
            "o subpregunta de la consulta del usuario. Marca 'False' ÚNICAMENTE si los datos están totalmente desvinculados, "
            "erróneos o carecen de valor semántico para la consulta."
        )
    )
    explicacion: str = Field(
        description="Breve justificación del veredicto de relevancia."
    )

# Modelo estructurado para evaluar la relevancia de los resultados
relevance_grader = model.with_structured_output(RelevanceGrade)

#4 DECLARAR HERRAMIENTAS
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="basic",
    include_answer=False,
    include_raw_content=False
)

extract_tool = TavilyExtract(
    extract_depth="basic",
    include_images=False
)

tools = [search_tool, extract_tool]
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

#6 HERRAMIENTA AUXILIAR: IMPRIMIR HERRAMIENTAS USADAS
def print_tool_message(tool_name: str, args: dict, content: str, grade: RelevanceGrade = None):
    print("\n" + "=" * 60)
    print(f"🛠️  [TOOL EXECUTION] -> {tool_name}")
    print(f"📥 Arguments: {args}")
    if grade:
        estado_eval = "✅ RELEVANTE" if grade.es_relevante else "❌ NO RELEVANTE"
        print(f"📋 Evaluador LLM (Pydantic): {estado_eval} -> {grade.explicacion}")
    print("-" * 60)
    print(f"📄 Output:\n{content}")
    print("=" * 60 + "\n")

#6.1 FILTRADO POR SCORE Y DESCARTE DE BÚSQUEDAS INFRUCTUOSAS
def sanitize_and_filter_payload(observation: any, min_score: float = 0.50, max_chars: int = 1500) -> tuple[str, bool]:
    """
    Filtra los resultados de Tavily con un score inferior a min_score.
    Si la búsqueda no devuelve nada relevante, se marca como infructuosa.
    """
    if isinstance(observation, list):
        filtered_results = []
        for idx, item in enumerate(observation, 1):
            if isinstance(item, dict):
                score = item.get("score", 1.0)
                # Filtrar resultados con score bajo
                if score < min_score:
                    continue
                
                title = item.get("title", "Sin título")
                url = item.get("url", "Sin URL")
                content = item.get("content", item.get("raw_content", ""))
                filtered_results.append(
                    f"[{idx}] {title} (Score: {score:.2f})\nURL: {url}\nTexto: {content}"
                )
            else:
                filtered_results.append(str(item))

        if not filtered_results:
            return (
                "⚠️ BÚSQUEDA INFRUCTUOSA: No se encontraron resultados web con suficiente relevancia (Score < 0.50). "
                "Intenta reformular la búsqueda usando otros términos clave.",
                False
            )

        cleaned_text = "\n\n".join(filtered_results)
    elif isinstance(observation, dict):
        cleaned_text = str(observation.get("results", observation))
    else:
        cleaned_text = str(observation)

    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars] + "\n... [Resultado truncado para ahorrar tokens]"
    
    return cleaned_text, True

#7 CALCULAR USO DE TOKENS
def calculate_token_usage(response, current_state: dict) -> dict:
    token_usage = response.usage_metadata or {}
    p_tokens = token_usage.get("input_tokens", 0)
    c_tokens = token_usage.get("output_tokens", 0)
    t_tokens = token_usage.get("total_tokens", p_tokens + c_tokens)

    return {
        "total_tokens": current_state.get("total_tokens", 0) + t_tokens,
        "prompt_tokens": current_state.get("prompt_tokens", 0) + p_tokens,
        "completion_tokens": current_state.get("completion_tokens", 0) + c_tokens,
    }

#7.1 REPORTE DE TOKENS Y HERRAMIENTAS
def print_token_and_tool_report(state: dict):
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

#8 NODO PRINCIPAL CON PROMPT CACHING (ANTHROPIC CACHE CONTROL)
def llm_call(state: dict):
    # Prompt caching mediante bloque con 'cache_control': {"type": "ephemeral"}
    system_prompt = SystemMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "You are a helpful assistant. Knowledge cutoff: July 2025. Current date: October 2026.\n"
                    "TOOLS GUIDE:\n"
                    "1. Use 'tavily_search' first to discover URLs and key snippets using short, keyword-focused queries.\n"
                    "2. If search snippets lack essential details, use 'tavily_extract' with specific URLs.\n"
                    "3. If a tool returns a 'BÚSQUEDA INFRUCTUOSA' or 'RESULTADO NO RELEVANTE' message, reformulate your query with alternative keywords."
                ),
                "cache_control": {"type": "ephemeral"}
            }
        ]
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

#9 NODO DE HERRAMIENTAS CON EVALUADOR DE PYDANTIC (FLEXIBILIZADO)
def tool_node(state: dict):
    result = []
    last_message = state["messages"][-1]
    added_tool_tokens = 0

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        raw_observation = tool.invoke(tool_call["args"])

        # 1. Filtrar resultados por score (>= 0.50) y desinfectar payload
        content_str, es_exitosa = sanitize_and_filter_payload(raw_observation, min_score=0.50, max_chars=1500)

        # 2. Evaluación de Relevancia del LLM usando Pydantic Schema si hubo resultados
        grade_result = None
        if es_exitosa:
            user_query = state["messages"][0].content if state["messages"] else ""
            try:
                # Prompt flexibilizado para que apruebe si responde al menos a un aspecto de la consulta
                grade_prompt = (
                    f"Consulta original del usuario: '{user_query}'\n\n"
                    f"Resultados de la herramienta:\n{content_str}\n\n"
                    f"INSTRUCCIÓN: Evalúa si los resultados aportan información útil o responden AL MENOS A UN ASPECTO "
                    f"o subpregunta del usuario. No exijas que responda a toda la consulta compleja de una sola vez."
                )
                grade_result = relevance_grader.invoke(grade_prompt)
                if not grade_result.es_relevante:
                    content_str = (
                        f"⚠️ RESULTADO NO RELEVANTE SEGÚN EL EVALUADOR LLM.\n"
                        f"Motivo: {grade_result.explicacion}\n"
                        f"Sugerencia: Reformula los términos de búsqueda."
                    )
            except Exception:
                pass

        added_tool_tokens += max(1, len(content_str) // 4)

        print_tool_message(
            tool_name=tool_call["name"],
            args=tool_call["args"],
            content=content_str,
            grade=grade_result
        )

        result.append(
            ToolMessage(
                content=content_str,
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": result,
        "tool_tokens": state.get("tool_tokens", 0) + added_tool_tokens
    }

#10 NODO FINAL
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

#11 CONDICIONAL Y GRAFO CON LÍMITE DE ITERACIONES (MAX_LLM_CALLS = 4)
MAX_LLM_CALLS = 4

def should_continue(state: MessagesState) -> Literal["tool_node", "report_node"]:
    messages = state["messages"]
    last_message = messages[-1]

    # Límite de seguridad para detener bucles de re-búsqueda excesivos
    if state.get("llm_calls", 0) >= MAX_LLM_CALLS:
        return "report_node"

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
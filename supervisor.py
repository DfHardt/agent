from dotenv import load_dotenv
import os
from datetime import datetime
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.tools import tool

import rich
from rich.console import Console

import tools as my_tools
from state import State
import config

load_dotenv()
console = config.console

def set_role_prompt(State, content: str):
    """
    Atualiza ou injeta o system prompt de função ('role_prompt') dentro do State['messages'].
    """
    updated = []
    replaced = False

    for msg in State["messages"]:
        if isinstance(msg, SystemMessage) and getattr(msg, "id", None) == "role_prompt":
            updated.append(SystemMessage(content=content, id="role_prompt"))
            replaced = True
        else:
            updated.append(msg)

    if not replaced:
        updated.insert(0, SystemMessage(content=content, id="role_prompt"))

    State["messages"] = updated
    return State

class StoreChatInfo:
    def __init__(self):
        self.user_data = {
            "nome_completo": None,
            "numero_matricula": None,
            "unidade": None,
            "modo": None,
        }

    def store_info(self, info: dict):
        for key in self.user_data:
            if key in info:
                self.user_data[key] = info[key]
        console.print(self.info)
    
    def get_info(self):
        return self.user_data
    
    def is_complete(self):
        return all(value is not None and value != "" for value in self.user_data.values())

stored_infos = StoreChatInfo()

@tool(
        description='''
Registra as informações completas de uma interação dissertativa realizada pelo estudante.

Essa função deve ser utilizada para salvar as seguintes informações:
    - Pergunta elaborada pelo monitor
    - Resposta do aluno (texto dissertativo)
    - Feedback do monitor

Retorna um dicionário estruturado com todas as informações coletadas.

Parâmetros:
- pergunta (str): Pergunta crítica formulada com base na unidade temática.
- resposta (str): Texto dissertativo escrito e confirmado pelo estudante.
- feedback (str): Comentário formativo elaborado pelo monitor com base na resposta.
'''
)
def register_chat_info(
pergunta: str | None = None,
resposta: str | None = None,
feedback: str | None = None,
) -> dict:
    user_data = stored_infos.get_info()
    chat_data = {
        'user_data': user_data,
        'pergunta': pergunta,
        'resposta': resposta,
        'feedback': feedback,
        'data': datetime.now()
    }
    return chat_data

@tool(
description = """
Armazena informações fornecidas sobre o usuário no dicionário `user_data`.

Esta função deve ser chamada apenas quando o agente receber uma das seguintes informações:
    a. nome completo do usuário;
    b. número de matrícula do usuário;
    c. unidade temática;
    d. modo de interação escolhido.

Args:
    info (dict): Dicionário contendo uma ou mais das seguintes chaves:
        - "nome_completo": str | None
        - "numero_matricula": str | int | None
        - "unidade": str | None
        - "modo": str | None

Exemplo de uso:
    store_info({
        "nome_completo": "João da Silva",
        "unidade": "Campus Norte"
    })

Notas:
    - Incluir as chaves no dicionário de entrada apenas se a informação não for recebida, caso contrário não inclua no dicionário de entrada.
    - A função sobrescreve valores antigos apenas para as chaves fornecidas.
"""
)
def storage_tool(
nome_completo: str | None = None,
numero_matricula: str | int | None = None,
unidade: str | None = None,
modo: str | None = None,
):
    info = {
        'nome_completo': nome_completo,
        'numero_matricula': numero_matricula,
        'unidade' : unidade,
        'modo': modo
    }
    stored_infos.store_info(info)

llm_router = config.llm.bind_tools(
    tools = [storage_tool]
    )

llm_mode_1 = config.llm.bind_tools(
    tools = [register_chat_info]
)

def supervisor_node(State: State):
    set_role_prompt(State, content = config.supervisor_prompt)
    result = llm_router.invoke(State['messages'])
    return {'messages': [result]}

#todo: função para contar o número de linhas e caracteres válidos;
#todo: reescrever o prompt qnd fizer a função do email
def mode_1(State: State):
    set_role_prompt(State, content = config.mode_1_sys_prompt)
    result = llm_mode_1.invoke(State['messages'])
    return {'messages': [result]}

tool_node = my_tools.BasicToolNode

def duvidas_conceituais(State: State):
    console.print('Modo 2: Dúvidas Conceituais')

def dúvidas_plano_de_ensino(State: State):
    console.print('Modo 3: Dúvidas sobre o Plano de Ensino')

def mode_router(State: State):
    """
    Use na contitional_edge para rotear o fluxo do agente para o modo de interação escolhido
    quando todas as informações obrigatórias tiverem sido registradas.
    """
    if isinstance(State, list):
        ai_message = State[-1]
    elif messages := State.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {State}")
    if stored_infos.user_data['modo'] is None:
        return '0'
    if any(x in stored_infos.user_data['modo'] for x in ['1', 'dissertativa']):
        return '1'
    if any(x in stored_infos.user_data['modo'] for x in ['2', 'conceitual']):
        return '2'
    if any(x in stored_infos.user_data['modo'] for x in ['3', 'plano']):
        return '3'

graph_builder = StateGraph(State)
graph_builder.add_node("router", supervisor_node)
graph_builder.add_node('router_tool_node', tool_node)
graph_builder.add_node("mode_1", mode_1)
graph_builder.add_node('mode_1_tool_node', tool_node)
graph_builder.add_node("mode_2", duvidas_conceituais)
graph_builder.add_node("mode_3", dúvidas_plano_de_ensino)

graph_builder.add_conditional_edges(
    "router",
    mode_router,
    {'0': END,'1':'mode_1', '2': 'mode_2', '3': 'mode_3'}
)

graph_builder.add_conditional_edges(
    'router',
    my_tools.route_tools,
    {'no tool_call': END, 'tools': 'router_tool_node'}
)

graph_builder.add_conditional_edges(
    'mode_1',
    my_tools.route_tools,
    {'no tool_call': END, 'tools': 'mode_1_tool_node'}
)

graph_builder.add_edge(START, 'router')
graph_builder.add_edge("mode_1", END)
graph_builder.add_edge('mode_1_tool_node', 'mode_1')
graph_builder.add_edge("mode_2", END)
graph_builder.add_edge("mode_3", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer = memory)

png_bytes = graph.get_graph().draw_mermaid_png()
with open("supervisor_graph.png", "wb") as f:
    f.write(png_bytes)

prompt_map = {
    'supervisor': config.supervisor_prompt,
    'mode_1': config.mode_1_sys_prompt,
}
while True:
    user_input = input("User: ")

    if user_input.lower() in ['q', 'quit', 'exit']:
        break
    
    prompt = {"messages": [
                SystemMessage(content = config.system_prompt, id = 'system_prompt'),
                HumanMessage(content = user_input)
            ]}
    result = graph.invoke(prompt, config = {'configurable': {'thread_id': '1'}})
    last_message = result["messages"][-1]

    console.print(f"[bold green3]Assistant:[/] [grey78]{last_message.content}[/]")
    
    

    
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from pinecone import Pinecone

import config as cfg

model = cfg.llm
console = cfg.console

pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
index = pc.Index("celso-db")

#. Tools

class StoreChatInfo:
    def __init__(self):
        self.user_data = {
            "nome_completo": None,
            "numero_matricula": None,
            "unidade": None,
            "modo": None,
        }

        self.email_content = {
            'user_data': self.user_data,
            'pergunta': None,
            'resposta': None,
            'feedback': None,
            'data': datetime.now().strftime('%d/%m/%Y %H:%M')
        }   

    def store_user_data(self, info: dict):
        for key in self.user_data:
            if key in info:
                self.user_data[key] = info[key]
    
    def build_email(self, email_body: dict):
        for key in self.email_content:
            if key in email_body:
                self.email_content[key] = email_body[key]
    
    def get_info(self):
        return self.user_data
    
    def is_complete(self):
        return all(value is not None and value != "" for value in self.user_data.values())

stored_infos = StoreChatInfo()


def storage_tool(
nome_completo:    str | None = None,
numero_matricula: str | int | None = None,
unidade:          str | None = None,
modo:             str | None = None,
):
    
    info = {
        'nome_completo': nome_completo,
        'numero_matricula': numero_matricula,
        'unidade' : unidade,
        'modo': modo
    }
    stored_infos.store_user_data(info = info)

    return info
storage_tool.__doc__ = cfg.Docstrings.storage_tool

def register_chat_info(
pergunta: str | None = None,
resposta: str | None = None,
feedback: str | None = None,
):
    chat_data = {
        'pergunta': pergunta,
        'resposta': resposta,
        'feedback': feedback,
    }
    stored_infos.build_email(email_body = chat_data)
    
    return chat_data
register_chat_info.__doc__ = cfg.Docstrings.register_chat_info

def send_email(registered_info: dict = stored_infos.email_content):
    msg = EmailMessage()
    msg['Subject'] = 'Registro de Interação - Monitor História Econômica II'
    msg['From'] = 'Celso Bot'
    msg['To'] = 'dfghardt@gmail.com'
    msg.set_content(
f'''
Nome: {registered_info['user_data']['nome_completo']};
Matrícula: {registered_info['user_data']['numero_matricula']};
Unidade temática: {registered_info['user_data']['unidade']};
Pergunta orientadora: {registered_info['pergunta']};
Resposta do estudante: 

{registered_info['resposta']}

Feedback do monitor: {registered_info['feedback']};
Data: {registered_info['data']};
'''
    )

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('dfghardt@gmail.com', cfg.gmail_pwd)
        smtp.send_message(msg)
send_email.__doc__ = cfg.Docstrings.send_email

def mode_3_retriever(query: str):
    output = '\n\n'
    results = index.search(
        namespace='Plano_de_Ensino_Historia_Economica_II_UFV',
        query={
            'inputs': {'text': query},
            'top_k': 3
        }
    )

    for chunk in results['result']['hits'][0]['fields']['chunk_text']:
        output += chunk

    return output
mode_3_retriever.__doc__ = cfg.Docstrings.mode_3_retriever

#. /Tools

#. handoff tools

# Agente 0: Coletor de Dados Iniciais
transfer_to_data_collector = create_handoff_tool(
    agent_name="data_collector",
    description="Use para o primeiro contato com o estudante. Este assistente é responsável por coletar os dados iniciais obrigatórios: nome completo, número de matrícula e a Unidade Temática de interesse."
)

# Agente 1: Modo Dissertativo (Registrado)
transfer_to_mode_1 = create_handoff_tool(
    agent_name="corretor_de_ensaios",
    description="Use quando o estudante escolher a interação dissertativa (Modo 1). Este assistente propõe uma pergunta crítica, analisa a resposta do estudante, fornece feedback formativo e, ao final, registra a interação completa para envio."
)

# Agente 2: Dúvidas e Exercícios (Não Registrado)
transfer_to_mode_2 = create_handoff_tool(
    agent_name="responder_duvidas",
    description="Use quando o estudante quiser esclarecer dúvidas conceituais sobre a matéria ou solicitar exercícios (Modo 2). As interações neste modo não são registradas."
)

# Agente 3: Dúvidas sobre o Plano de Ensino (Não Registrado)
transfer_to_mode_3 = create_handoff_tool(
    agent_name="agente_plano_de_ensino",
    description="Use exclusivamente quando o estudante tiver dúvidas sobre o plano de ensino (Modo 3), como datas das aulas, textos base e cronograma. As interações neste modo não são registradas."
)

#. /handoff tools

#. ReAct Agents

data_collector = create_react_agent(
    model=model,
    tools=[storage_tool, transfer_to_mode_1, transfer_to_mode_2, transfer_to_mode_3],
    prompt= cfg.SwarmArchitecturePrompts.data_collector_prompt,
    name="data_collector"
)

mode_1 = create_react_agent(
    model=model,
    tools=[send_email, register_chat_info, transfer_to_data_collector, transfer_to_mode_2, transfer_to_mode_3],
    prompt= cfg.SwarmArchitecturePrompts.mode_1_prompt,
    name="corretor_de_ensaios"
)

mode_2 = create_react_agent(
    model=model,
    tools=[transfer_to_data_collector, transfer_to_mode_1, transfer_to_mode_3],
    prompt= cfg.SwarmArchitecturePrompts.mode_2_prompt,
    name="responder_duvidas"
)

mode_3 = create_react_agent(
    model=model,
    tools=[transfer_to_data_collector, transfer_to_mode_1, transfer_to_mode_2, mode_3_retriever],
    prompt= cfg.SwarmArchitecturePrompts.mode_3_prompt,
    name="agente_plano_de_ensino"
)

#. /ReAct Agents

checkpointer = MemorySaver()

builder = create_swarm(
    [data_collector, mode_1, mode_2, mode_3], default_active_agent="data_collector"
)

graph = builder.compile(checkpointer=checkpointer)

png_bytes = graph.get_graph().draw_mermaid_png()
with open("supervisor_graph.png", "wb") as f:
    f.write(png_bytes)

config = {"configurable": {"thread_id": "1"}}

def print_stream(stream):
    for ns, update in stream:
        for node, node_updates in update.items():
            if node_updates is None:
                continue

            if isinstance(node_updates, (dict, tuple)):
                node_updates_list = [node_updates]
            elif isinstance(node_updates, list):
                node_updates_list = node_updates
            else:
                raise ValueError(node_updates)

            for node_updates in node_updates_list:
                if isinstance(node_updates, tuple):
                    continue
                messages_key = next(
                    (k for k in node_updates.keys() if "messages" in k), None
                )
                if messages_key is not None:
                    node_updates[messages_key][-1].pretty_print()
                else:
                    pass

def ApplicationLoop():
    while True:
        user_input = console.input("[bold magenta]User: [/]")

        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("[green3]Goodbye![/green3]")
            break

        prompt = {"messages": [
                SystemMessage(content = cfg.system_prompt, id = 'sys_prompt'),
                HumanMessage(content = user_input)
            ]}

        result = graph.invoke(prompt, config = {'configurable': {'thread_id': '1'}, "recursion_limit": 10})
        last_message = result["messages"][-1]

        console.print(f"[bold green3]Celso:[/] [grey78]{last_message.content}[/]")

ApplicationLoop()
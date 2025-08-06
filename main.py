from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

from typing import Annotated, List
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

import rich
from rich.console import Console

import tools as my_tools
from state import State
import config as cfg

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
console = Console()


#_ Models

data_collector_tools = [my_tools.storage_tool, my_tools.transfer_to_corretor_de_ensaios, my_tools.transfer_to_responder_duvidas, my_tools.transfer_to_agente_plano_de_ensino]
data_collector_model = cfg.llm.bind_tools(
    tools = data_collector_tools,
    parallel_tool_calls = False
)
mode_1_tools = [my_tools.storage_tool, my_tools.register_chat_info, my_tools.transfer_to_data_collector, my_tools.transfer_to_responder_duvidas, my_tools.transfer_to_agente_plano_de_ensino, my_tools.transfer_to_email_sender]
mode_1_model = cfg.llm.bind_tools(
    tools=mode_1_tools,
    parallel_tool_calls=False
)

mode_2_tools = [my_tools.transfer_to_data_collector, my_tools.transfer_to_corretor_de_ensaios, my_tools.transfer_to_agente_plano_de_ensino]
mode_2_model = cfg.llm.bind_tools(
    tools=mode_2_tools,
    parallel_tool_calls=False
)

mode_3_tools = [my_tools.transfer_to_data_collector, my_tools.transfer_to_corretor_de_ensaios, my_tools.transfer_to_responder_duvidas, my_tools.mode_3_retriever]
mode_3_model = cfg.llm.bind_tools(
    tools=mode_3_tools,
    parallel_tool_calls=False
)

email_sender_tools = [my_tools.get_user_data, my_tools.storage_tool, my_tools.transfer_to_corretor_de_ensaios, my_tools.transfer_to_responder_duvidas, my_tools.transfer_to_agente_plano_de_ensino, my_tools.send_email]
email_sender_model = cfg.llm.bind_tools(
    tools=email_sender_tools,
    parallel_tool_calls=False
)

#. ReAct Agents

data_collector = create_react_agent(
    model=data_collector_model,
    tools=data_collector_tools,
    prompt= cfg.Prompts.data_collector_prompt,
    name="data_collector",
)

mode_1 = create_react_agent(
    model=mode_1_model,
    tools=mode_1_tools,
    prompt= cfg.Prompts.mode_1_prompt,
    name="corretor_de_ensaios",
)

mode_2 = create_react_agent(
    model = mode_2_model,
    tools = mode_2_tools,
    prompt = cfg.Prompts.mode_2_prompt,
    name = "responder_duvidas",
)

mode_3 = create_react_agent(
    model=mode_3_model,
    tools=mode_3_tools,
    prompt= cfg.Prompts.mode_3_prompt,
    name="agente_plano_de_ensino",
)

email_sender = create_react_agent(
    model = email_sender_model,
    tools = email_sender_tools,
    prompt = cfg.Prompts.email_sender_prompt,
    name  = "email_sender",
)
#_ Graph

checkpointer = MemorySaver()

builder = create_swarm(
    [data_collector, mode_1, mode_2, mode_3, email_sender], 
    default_active_agent="data_collector",
)

graph = builder.compile(checkpointer=checkpointer)

png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

def ApplicationLoop():
    while True:
        user_input = console.input("[bold magenta]User: [/]")

        if user_input.lower() in ["quit", "exit", "q"]:
            rich.print("[green3]Goodbye![/green3]")
            break

        prompt = {"messages": [
                SystemMessage(content = cfg.Prompts.system_prompt, id = 'sys_prompt'),
                HumanMessage(content = user_input)
            ]}

        result = graph.invoke(prompt, config = {'configurable': {'thread_id': '1'}})
        last_message = result["messages"][-1]

        console.print(f"[bold green3]Assistant:[/] [grey78]{last_message.content}[/]")
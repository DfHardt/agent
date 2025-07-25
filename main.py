from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import rich
from rich.console import Console

import tools as my_tools
from state import State
import config

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
console = Console()

#. Nodes

llm = ChatGroq(model = "llama-3.3-70b-versatile").bind_tools(tools = my_tools.tools, tool_choice = "auto")

def chatbot(State: State):
    try:
        result = llm.invoke(State['messages'])
        with open('log.txt', 'a') as log:
            log.write(f'Node: chatbot\n\n\
State: {State['messages']}\n\n\
Output: {result}\n\n')
            log.close()
        return {'messages': [result]}
    except Exception as err:
        print('Erro no chatbot node:', err)
        return {"messages": [{"role": "system", "content": "Ocorreu um erro:, tente novamente com outro prompt."}]}

tool_node = my_tools.BasicToolNode(tools = my_tools.tools)

#. Graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node('tool_node', tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    my_tools.route_tools,
    {"tools": "tool_node", 'no tool_call': END},
)

graph_builder.add_edge('tool_node', 'chatbot')
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer = memory)

png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

def ApplicationStreamLoop():

    def stream_graph_updates(user_input: str):

        prompt = {"messages": [
            {"role": "system", "content": "Responda sempre em português do Brasil. Seu nome é Celso e você é um assistente submisso e gentil."},
            {"role": "user", "content": user_input}
        ]}

        for event in graph.stream(prompt, config = {'configurable': {'thread_id': '5'}}):
            for value in event.values():
                console.print(f"[bold green3]Assistant:[/] [grey78]{value['messages'][-1].content}[/]")

    while True:
        user_input = console.input("[bold magenta]User: [/]")
        if user_input.lower() in ["quit", "exit", "q"]:
            rich.print("[green3]Goodbye![/green3]")
            break
        stream_graph_updates(user_input)

def ApplicationLoop():
    with open('log.txt', 'a') as log:
        log.write('Início da run\n\n')
        log.close()

    while True:
        user_input = console.input("[bold magenta]User: [/]")

        if user_input.lower() in ["quit", "exit", "q"]:
            rich.print("[green3]Goodbye![/green3]")
            break

        prompt = {"messages": [
                SystemMessage(content = config.system_prompt, id = 'sys_prompt'),
                HumanMessage(content = user_input)
            ]}

        result = graph.invoke(prompt, config = {'configurable': {'thread_id': '1'}})
        last_message = result["messages"][-1]

        console.print(f"[bold green3]Assistant:[/] [grey78]{last_message.content}[/]")


ApplicationLoop()
#ApplicationStreamLoop()































'''
trash 

graph_builder.add_conditional_edges(
    "chatbot",
    my_tools.route_tools,
    {"tools": "tools", END: END},
)
'''
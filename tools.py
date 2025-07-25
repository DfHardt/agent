import json, os, math, inspect, requests, random, dotenv, re

from typing import Callable

from langchain.agents import Tool
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, FunctionMessage, BaseMessage, AnyMessage

from state import State
import rag
import config

dotenv.load_dotenv()

tvly_api = os.getenv('tvly_api')
os.environ["TAVILY_API_KEY"] = tvly_api
openweather_api = os.getenv('openweather_api')


#. Tool Node (Function to run the tools)
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            return {"messages": outputs}

#. Tool router   
def route_tools(State: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(State, list):
        ai_message = State[-1]
    elif messages := State.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {State}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return 'no tool_call'

#__ RAG

@tool
def retriever(query: str):
    '''
    Carrega um vectorstore do banco de dados.
    retorna o resultado da busca semântica do retriever.
    Args:
        query (str): Assunto a ser pesquisado no documento.
    
    Você tem acesso aos seguintes documentos:

    Plano_de_Ensino_Historia_Economica_II_UFV: Contém a ementa e o cronograma da disciplina,
    Programa_Analitico-Historia_Economica_II: Contém o objetivo da disciplina,
    Unidade_I
    '''
    emb = OllamaEmbeddings(model = 'granite-embedding')
    folder = r'rag_resources\main_vector_db\main_db'

    vectorstore = FAISS.load_local(folder, embeddings = emb, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    
    docs = retriever.invoke(query)
    return '\n\n'.join(doc.page_content for doc in docs)

@tool
def pinecone_retriever(query: str):
    '''
    Utilize essa ferramenta para consultar o banco de dados da disciplina História Econômica 2.
    Faz a pesquisa no banco de dados a partir da entrada do usuário.
    Retorna os resultados com maior similaridade semantica em relação ao input.
    Args:
        query (str): Entrada do usuário
    Return:
        Contexto gerado a partir da busca por similaridade semantica no banco de dados.
    '''

    return rag.pinecone_retriever(query) + '\n\n'

tools = [
    pinecone_retriever
]
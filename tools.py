import json, os, math, inspect, requests, random, dotenv

from typing import Callable

from langchain.agents import Tool
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from langgraph.graph import StateGraph, START, END

from state import State
import rag

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
        with open('log.txt', 'a') as log:
            log.write(f'Node: tool_node\n\n\
ai_message: {message}\n\n\
Output: {outputs}\n\n')
            log.close()
        return {"messages": outputs}

#. Tool router   
def route_tools(State: State,):
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

#__ math tool
@tool
def sums_two_int(a: int | float, b:int | float) -> int | float:
    """
    sums two integers
    you must use it when asked two perform adding operation of two integers.
    args: a: int, b: int
    returns a+b
    """
    return a+b

@tool
def subtraction(a: int | float, b: int | float) -> int | float:
    """
    Subtrai o valor b do valor a.

    Argumentos:
    a (int | float): O minuendo, número do qual será subtraído outro número.
    b (int | float): O subtraendo, número que será subtraído do minuendo.

    Retorna:
    int | float: O resultado da subtração (a - b).

    Quando usar:
    Use esta função para calcular a diferença entre dois números inteiros ou reais.
    """
    return a-b

@tool
def multiplication(a: int | float, b: int | float) -> int | float:
    """
    Multiplica dois números, a e b.

    Argumentos:
    a (int | float): O primeiro fator da multiplicação.
    b (int | float): O segundo fator da multiplicação.

    Retorna:
    int | float: O resultado da multiplicação (a * b).

    Quando usar:
    Use esta função para calcular o produto de dois números inteiros ou reais.
    """
    return a*b

@tool
def division(a: int | float, b: int | float) -> int | float:
    """
    Divide o número a pelo número b.

    Argumentos:
    a (int | float): O numerador.
    b (int | float): O denominador. Deve ser diferente de zero.

    Retorna:
    int | float: O resultado da divisão (a / b).

    Quando usar:
    Use esta função para dividir dois números inteiros ou reais. 
    Certifique-se de que o denominador (b) não seja zero para evitar erro.
    """
    return a/b

@tool
def factorial(a: int) -> int:
    """
    Calcula o fatorial de um número inteiro não negativo.

    Argumentos:
    a (int): Número inteiro não negativo para calcular o fatorial.

    Retorna:
    int: O fatorial de a (a!).

    Quando usar:
    Use esta função para calcular o fatorial de um número inteiro não negativo.
    """
    return math.factorial(a)

#__ Climate tools
#todo criar mais uma função com outra api para atuar em redundância

nome_para_id = {}

with open(r'city.list.json\city.list.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)
    for cidade in dados:
        nome = cidade['name'].strip().lower()
        nome_para_id[nome] = cidade['id']

@tool
def weather_request(city_name: str | int):
    """
    Consulta as condições climáticas atuais de uma cidade usando sua ID no OpenWeatherMap.

    Argumentos:
    city_name (str | int): O Inome da cidade.

    Retorna:
    dict: Um dicionário com os dados do clima atual retornados pela API, incluindo temperatura, umidade, condições climáticas, etc.

    Quando usar:
    Use esta função para obter informações meteorológicas em tempo real de uma cidade específica a partir do seu nome. 
    """
    city_id = nome_para_id.get(city_name)
    request = requests.get(f'https://api.openweathermap.org/data/2.5/weather?id={city_id}&appid={openweather_api}')
    return request.json()

#__ Search tools

tvly_tool = TavilySearch(max_results = 2)

#__ Utility tools

@tool
def rng(a: int, b: int, n: int | None = None):
    """
    Gera números inteiros aleatórios no intervalo [a, b].

    Argumentos:
    a (int): Valor mínimo do intervalo.
    b (int): Valor máximo do intervalo.
    n (int | None, opcional): Quantidade de números a gerar. 
        Se não for fornecido, retorna apenas um número aleatório.

    Retorna:
    int | tuple[int, ...]: Um único número aleatório se n for None,
        ou uma tupla com n números aleatórios se n for fornecido.

    Quando usar:
    Use esta função para gerar números inteiros aleatórios dentro de um intervalo,
    seja para obter um único valor ou uma sequência de valores.
    """

    if n == None:
        return random.randint(a, b)
    
    aux = []
    for i in range(n):
        aux.append(random.randint(a,b))
    return tuple(aux)
        
#__ RAG

@tool
def retriever(file: str, query: str):
    '''
    Carrega um vectorstore do documento a ser pesquisado.
    retorna o vectorstore como retriever pronto para ser pesquisado.
    Args:
        file (str): Nome do documento que vai ser pesquisado.
        query (str): Assunto a ser pesquisado no documento.
    '''
    emb = OllamaEmbeddings(model = 'granite-embedding')

    if ' ' in file or '.' in file:
        file = file.rstrip('.pdf').replace(' ', '_').replace('.', '_')
    load_path = os.path.join('embed_text', file)

    vectorstore = FAISS.load_local(load_path, embeddings = emb, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    
    docs = retriever.invoke(query)
    return '\n\n'.join(doc.page_content for doc in docs)

#pdf_retriever = rag.pdf_retriever
#retriever_tool = rag.retriever_tool

tools = [
    tvly_tool, 
    sums_two_int,
    subtraction,
    multiplication,
    division,
    factorial,
    #weather_request,
    rng,
    #pdf_retriever,
    #rag.retrieve_pdf_content
    retriever
]
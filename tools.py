import json, os, dotenv
import smtplib
from email.message import EmailMessage

from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import ToolMessage
from langgraph_swarm import create_handoff_tool
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState

from pinecone import Pinecone

from state import State
import rag
import config as cfg

dotenv.load_dotenv()

tvly_api = os.getenv('tvly_api')
os.environ["TAVILY_API_KEY"] = tvly_api
openweather_api = os.getenv('openweather_api')

pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
index = pc.Index("celso-db")


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

#_ RAG
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

#_ Captura de dados

stored_infos = cfg.stored_info

@tool(description=cfg.ToolDocstrings.storage_tool)
def storage_tool(
    state: Annotated[Any, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    nome_completo: str | None = None,
    numero_matricula: str | int | None = None,
    unidade: str | None = None,
    modo: str | None = None,
):

    info = {
        "nome_completo": nome_completo,
        "numero_matricula": numero_matricula,
        "unidade": unidade,
        "modo": modo
    }

    # Armazena os dados (sua lógica)
    stored_infos.store_user_data(info=info)
    
    # Cria manualmente o ToolMessage (como fazia antes no tool node)
    return 'Informação registrada com sucesso'


#captura os dados da conversa
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
    
    return 'Informação registrada com sucesso, não esqueça de informar o feedback para o aluno'
register_chat_info.__doc__ = cfg.ToolDocstrings.register_chat_info

def is_data_missing():
    if stored_infos.is_data_missing() == []:
        return "Nenhuma informação está faltando, pode prosseguir com o envio do email"
    return stored_infos.is_data_missing()
is_data_missing.__doc__ = cfg.ToolDocstrings.is_data_missing

def send_email():
    msg = EmailMessage()
    msg['Subject'] = 'Registro de Interação - Monitor História Econômica II'
    msg['From'] = 'Celso Bot'
    msg['To'] = 'dfghardt@gmail.com'

    registered_info = stored_infos.email_content

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
send_email.__doc__ = cfg.ToolDocstrings.send_email

def mode_1_retriever(query: str):
    output = '\n\n'
    results = index.search(
        namespace="unified-db",
        query = {
            'inputs': {'text': query},
            'top_k': 3
        }
    )

    for chunk in results['result']['hits'][0]['fields']['chunk_text']:
        output += chunk

    return output
mode_1_retriever.__doc__ = cfg.ToolDocstrings.mode_1_retriever

def mode_2_retriever(query: str):
    output = '\n\n'
    results = index.search(
        namespace="unified-db",
        query = {
            'inputs': {'text': query},
            'top_k': 3
        }
    )

    for chunk in results['result']['hits'][0]['fields']['chunk_text']:
        output += chunk

    return output
mode_2_retriever.__doc__ = cfg.ToolDocstrings.mode_2_retriever

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
mode_3_retriever.__doc__ = cfg.ToolDocstrings.mode_3_retriever

#_ Handoff Tools

transfer_to_data_collector = create_handoff_tool(
    agent_name="data_collector",
    description="Use para o primeiro contato com o estudante. Este assistente é responsável por coletar os dados iniciais obrigatórios: nome completo, número de matrícula e a Unidade Temática de interesse."
)

# Agente 1: Modo Dissertativo (Registrado)
transfer_to_corretor_de_ensaios = create_handoff_tool(
    agent_name="corretor_de_ensaios",
    description="Use quando o estudante escolher a interação dissertativa (Modo 1). Este assistente propõe uma pergunta crítica, analisa a resposta do estudante, fornece feedback formativo e, ao final, registra a interação completa para envio.",
    name="transfer_to_corretor_de_ensaios"
)

# Agente 2: Dúvidas e Exercícios (Não Registrado)
transfer_to_responder_duvidas = create_handoff_tool(
    agent_name="responder_duvidas",
    description="Use quando o estudante quiser esclarecer dúvidas conceituais sobre a matéria ou solicitar exercícios (Modo 2). As interações neste modo não são registradas.",
    name="transfer_to_responder_duvidas"
)

# Agente 3: Dúvidas sobre o Plano de Ensino (Não Registrado)
transfer_to_agente_plano_de_ensino = create_handoff_tool(
    agent_name="agente_plano_de_ensino",
    description="Use exclusivamente quando o estudante tiver dúvidas sobre o plano de ensino (Modo 3), como datas das aulas, textos base e cronograma. As interações neste modo não são registradas.",
    name="transfer_to_agente_plano_de_ensino"
)

transfer_to_email_sender = create_handoff_tool(
    agent_name="email_sender",
    description="Use apenas depois de enviar o feedback da redação ao estudante e depois da confirmação do mesmo, ferramenta para transferir ao agente responsável pelo envio do email",
    name="transfer_to_email_sender"
)

tools = [
    pinecone_retriever
]
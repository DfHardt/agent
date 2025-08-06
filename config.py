from dotenv import load_dotenv
import os
import datetime

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
gmail_pwd = os.getenv('gmail')

from rich.console import Console
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

console = Console()
embedder = OllamaEmbeddings(model="granite-embedding")
llm = ChatGroq(model = "llama-3.3-70b-versatile")

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
            'data': datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
        }   
    
    def store_user_data(self, info: dict):
        for key in self.user_data:
            if key in info and info[key] is not None:
                self.user_data[key] = info[key]
    
    def build_email(self, email_body: dict):
        for key in self.email_content:
            if key in email_body and email_body[key] is not None:
                self.email_content[key] = email_body[key]
    
    def get_user_data(self):
        return self.user_data
    
    def is_corrected(self):
        return self.email_content['feedback'] is not None
    
stored_info = StoreChatInfo()


#_ Swarm Architecture Prompts
class Prompts:
    #_
    system_prompt = """
    Informações sobre você:
        1. Você é o Celso, monitor automatizado da disciplina História Econômica II (HIS 123) da Universidade Federal de Viçosa;
        2. O professor da disciplina é o André Luan Nunes Macedo;
        3. Você foi desenvolvido pela equipe NIAS-IA do Departamento de engenharia elétrica (DEL) em conjunto com o Prof. André.
    Regras:
        1. Nunca mencione o nome das ferramentas ao usuário.
    """

    #_
    data_collector_prompt = '''Sua função é iniciar a interação com o estudante e rotear o fluxo de agentes baseado no modo de interação escolhido.
    Você deve sempre apresentar o seguinte menu contendo os modos de interação:
        Modo 1 - Interação Dissertativa
        Modo 2 - Dúvidas e Exercícios
        Modo 3 - Dúvidas sobre o Plano de Ensino

    Quando o estudante informar o modo de interação escolhido você deve armazenar a informação utilizando a ferramenta storage_tool.
    Após o registro, você deve utilizar uma de suas handoff_tools para passar a vez ao agente responsável pelo modo de interação.'''
    
    #_
    mode_1_prompt = """
Seu papel e tarefa:
Elaborar e acompanhar interações dissertativas, nas quais o estudante responde a uma pergunta crítica com um texto próprio e você fornece feedback formativo;
Você deve armazenar a pergunta crítica, a resposta do estudante e o seu feedback utilizando a ferramenta register_chat_info

Siga rigorosamente as instruções abaixo:
1. Inicie a interação informando que este é o Modo 1 - Interação Dissertativa Registrada.

2. Solicite a unidade na qual o estudante está interessado (ofereça a lista abaixo)
    Lista de unidades:
        - Unidade 1: Instituições, dependência de trajetória e história econômica
        - Unidade 2: Economia européia e economia mundo no século XIX
        - Unidade 3: A emergência dos grandes conglomerados industriais e a segunda revolução industrial
        - Unidade 4: Democracia, socialismo e capitalismo
        - Unidade 5: A grande depressão nos Estados Unidos e na Europa: causas e desdobramentos
        - Unidade 6: Fordismo, regulação e Welfare State. Mudanças institucionais no pós-guerra: Bretton Woods, Plano Marshall e CEE
        - Unidade 7: Centro e periferia no capitalismo contemporâneo
        - Unidade 8: A crise dos anos 70 e os novos modelos de organização produtiva

2. Proponha uma pergunta crítica relacionada ao tema da unidade escolhida.
3. Solicite que o estudante responda com um texto dissertativo de 3 parágrafos, totalizando aproximadamente 12 linhas (cerca de 900 caracteres com espaços), escrito com suas próprias palavras.
4. Informe que o uso de outras ferramentas de IA pode prejudicar o aprendizado e que o objetivo da atividade é desenvolver a escrita crítica.

5. Ao receber o texto do usuário, realize esses dois processos:
    - 5.1. Informe ao aluno um feedback formativo sobre o seu texto;
    - 5.2. Pergunte se ele deseja fazer alguma alteração no texto dele e se ele deseja enviar o relatório da conversa ao professor.

Regras:
Você não deve simular, completar ou sugerir respostas pelo estudante, nem mesmo parcialmente. Ele deve escrevê-la integralmente.
"""

    #_
    mode_2_prompt = """ 
Seu papel e tarefa:
Você é o monitor responsável por esclarecer dúvidas conceituais sobre os temas da disciplina e propor exercícios, como questões de múltipla escolha ou verdadeiro/falso;.

Caso o estudante solicite esclarecimento de dúvidas sobre um tema específico, responda SOMENTE com base na bibliografia da disciplina de forma clara e objetiva.

Caso o estudante solicite exercícios, você pode elaborar questões fechadas (múltipla escolha, verdadeiro/falso, correspondência etc.) com base no conteúdo estudado.
"""

    #_
    mode_3_prompt = """
Seu papel e tarefa:
Você é o assistente responsável por esclarecer dúvidas sobre o plano de ensino, como as datas das aulas e dos textos base.
Caso o estudante solicite esclarecimento de dúvidas sobre o plano de ensino, faça a leitura do documento "Plano_de_Ensino_História_Econômica_II_UFV.docx" que está na sua base de conhecimento para responder. Você deve se ater EXCLUSIVAMENTE às informações deste documento.
"""

    #_
    email_sender_prompt = """
Seu papel e tarefa:
Você é o agente responsável por coletar as informações pessoais do estudante e enviar o email contendo os dados da interação dissertativa para o professor.

Siga rigorosamente as instruções abaixo: 
1. Verifique o registro das informações necessárias para o envio do email;
    - Se houver alguma informação faltando, solicite que o usuário a envie novamente.
    - Importante: Armazene as informações coletadas utilizando a ferramenta storage_tool.
    
2. Após o estudante fornecer todas as informações obrigatórias, envie o email.
"""


class ToolDocstrings:
    register_chat_info = '''
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

    storage_tool = """
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
    storage_tool({
        "nome_completo": "João da Silva",
        "unidade": "Campus Norte"
    })

Notas:
    - Incluir as chaves no dicionário de entrada apenas se a informação não for recebida, caso contrário não inclua no dicionário de entrada.
    - A função sobrescreve valores antigos apenas para as chaves fornecidas.
"""

    send_email = """
Envia os dados obrigatórios coletados no modo de interação 1 para o e-mail do professor André.
Não recebe argumentos. Deve ser chamada apenas após o usuário confirmar o envio.
Exemplo de uso:
send_email()
"""

    mode_3_retriever = '''
Use essa ferramenta para acessar o Plano de Ensino de História Econômica 2.
Args:
    Query (str): Conteúdo a ser pesquisado no documento.
'''

    get_user_data = '''
Verifica os dados armazenados necessários para o envio do email.
'''

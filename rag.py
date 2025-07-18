import os
import numpy as np

import faiss
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

embedding_model = 'granite-embedding'

#__ RAG
"""
class PDFRetriever:
    def __init__(self, file_path: str, embedding_model: str = embedding_model):
        self.file_path = file_path
        self.embedding_model = embedding_model
        self.pdf_list = [file for file in os.listdir(file_path)
                         if os.path.isfile(os.path.join(file_path, file))
                         and file.endswith('.pdf')]

    def load(self):
        docs = [PyPDFLoader(os.path.join(self.file_path, pdf)).load() for pdf in self.pdf_list]
        return [doc for sublist in docs for doc in sublist] 
    
    def split(self, documents):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 500, chunk_overlap = 100
        )

        return text_splitter.split_documents(documents)
    
    def retriever(self, pdf_name: str = None):
        '''
        Cria e retorna um retriever vetorizado para um PDF específico.

        Args:
            pdf_name (str): Nome do arquivo PDF a ser carregado e vetorizar.

        Returns:
            BaseRetriever: objeto retriever para busca vetorizada no conteúdo do PDF.
        '''
        try:
            if pdf_name:
                docs = PyPDFLoader(os.path.join(self.file_path, pdf_name)).load()
                splitted = self.split(docs)
                vector_store = InMemoryVectorStore.from_documents(
                    documents=splitted,
                    embedding=OllamaEmbeddings(model=self.embedding_model)
                )
                return vector_store.as_retriever()

            vector_store = InMemoryVectorStore.from_documents(
                documents = self.split(self.load()),
                embedding = OllamaEmbeddings(model=embedding_model)
            )
            return vector_store.as_retriever()
        except:
            vector_store = InMemoryVectorStore.from_documents(
                documents = self.split(self.load()),
                embedding = OllamaEmbeddings(model=embedding_model)
            )
            return vector_store.as_retriever()

pdf_retriever = PDFRetriever(file_path = 'pdfs', embedding_model = embedding_model)
pdf_retriever = create_retriever_tool(
    pdf_retriever.retriever(),
    'retrieve_pdf_content',
    f'''
    Busca e retorna informações a partir de PDFs.
    Você tem acesso aos seguintes documentos: {[file for file in os.listdir('pdfs') if file.endswith('.pdf')]}.
    Você deve utilizar essa ferramenta quando for perguntado sobre algum assunto abordado por esses documentos.
    Args:
    pdf_name (str): Nome do pdf a ser consultado.
    '''
)"""

class PDFRetriever:
    def __init__(self, file_path: str, embedding_model: str = embedding_model):
        self.file_path = file_path
        self.embedding_model = embedding_model
        self.pdf_list = [file for file in os.listdir(file_path)
                         if os.path.isfile(os.path.join(file_path, file))
                         and file.endswith('.pdf')]

    def load(self):
        docs = [PyPDFLoader(os.path.join(self.file_path, pdf)).load() for pdf in self.pdf_list]
        return [doc for sublist in docs for doc in sublist] 
    
    def split(self, documents):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = 500, chunk_overlap = 100
        )

        return text_splitter.split_documents(documents)
    
    def retriever(self, file_name: str = None):
        '''
        Cria e retorna um retriever vetorizado para um PDF específico.

        Args:
            file_name (str): Nome do arquivo vetorizado a ser carregado.

        Returns:
            BaseRetriever: objeto retriever para busca vetorizada no conteúdo do PDF.
        '''
        if file_name:
            vectorstore = FAISS.load_local(os.path.join(r'embed_text', file_name), embeddings=embedding_model)
            return vectorstore.as_retriever()
        print('file_name=none')
    
pdf_retriever = PDFRetriever(file_path = 'pdfs', embedding_model = embedding_model)


desc = f'''
    Busca e retorna informações a partir de PDFs vetorizados.
    Você tem acesso aos seguintes documentos: {[file for file in os.listdir('embed_text') if file.endswith('.npy')]}.
    Você deve utilizar essa ferramenta quando for perguntado sobre algum assunto abordado por esses documentos.
    Args:
    file_name (str): Nome do arquivo a ser consultado.
    query (str): Solicitação do usuário.
    '''
@tool(description=desc)
def retrieve_pdf_content(file_name: str, query: str):
    retriever = pdf_retriever.retriever(file_name)
    if retriever is None:
        return "Arquivo não encontrado ou file_name não especificado."
    docs = retriever.get_relevant_documents(query)
    return [d.page_content for d in docs]

@tool(description = desc)
def retrieve_from_loaded(file_name: str, query: str):
    db = FAISS.load_local(os.path.join('embed_text', file_name))
    retriever = VectorStoreRetriever(vectorstore=db)
    llm = llm = ChatGroq(model = "llama-3.3-70b-versatile")

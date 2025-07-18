from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

import tqdm
import os
import time
import rich
import numpy as np
import faiss
import json


embedder = OllamaEmbeddings(model="granite-embedding")
file_path = 'pdfs'

file_list = [file for file in os.listdir(file_path)
             if os.path.isfile(os.path.join(file_path, file))
             and file.endswith('.pdf')]

rich.print('[dark_red]Loading and embedding files...[/]')

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)

def embedder_fn():
    os.makedirs('embed_text', exist_ok=True)

    for file_name in file_list:
        start = time.time()
        rich.print(f'[dark_red]Começando embedding e indexação do {file_name}[/]')

        loader = PyPDFLoader(os.path.join(file_path, file_name))
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)

        all_embeddings = []
        for chunk in tqdm.tqdm(chunks, desc="Chunks embedding"):
            emb = embedder.embed_query(chunk.page_content)
            all_embeddings.append(emb)

        # Cria índice FAISS manualmente
        dimension = len(all_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(all_embeddings).astype('float32'))

        # Cria o FAISS Vectorstore com o índice e documentos
        vectorstore = FAISS(embedding_function=None)
        vectorstore.index = index
        vectorstore.docstore.add_documents(chunks)

        # Salva na pasta embed_text com nome limpo
        base_name = file_name.rstrip('.pdf').replace(' ', '_').replace('.', '_')
        save_path = os.path.join('embed_text', base_name)
        vectorstore.save_local(save_path)

        end = time.time()
        rich.print(f'[dark_red]Embedding e indexação finalizados. Duração: {end - start:.2f}s.')

#embedder_fn()

#__ Save metadata
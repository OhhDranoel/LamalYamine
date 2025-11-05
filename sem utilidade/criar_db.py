# criar_db.py
import shutil
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pasta_base = "base"
CAMINHO_DB = "db"

def criar_db():
    # 🔹 Remove DB antigo
    if os.path.exists(CAMINHO_DB):
        shutil.rmtree(CAMINHO_DB)
        print("🗑️ DB antigo removido")

    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(pasta_base, glob="*.pdf")
    documentos = carregador.load()
    print(f"📄 {len(documentos)} documentos carregados da pasta '{pasta_base}'")
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=500,       # tamanho menor para perguntas curtas
        chunk_overlap=50,     # pequeno overlap
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(f"✂️ {len(chunks)} chunks gerados")
    return chunks

def vetorizar_chunks(chunks):
    func = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, func, persist_directory=CAMINHO_DB)
    print("✅ Banco de dados criado")

if __name__ == "__main__":
    criar_db()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pasta_base = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(pasta_base, glob="*.pdf")
    documentos = carregador.load()
    return documentos
    
def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=1000,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks

def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
    print("Banco de dados criado")

criar_db()
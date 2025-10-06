# reindex.py
import os
from dotenv import load_dotenv
import re
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma


print("[DEBUG] Carregando variáveis de ambiente...")
load_dotenv()
print("[DEBUG] Variáveis de ambiente carregadas.")

CAMINHO_DB = "db"
ARQUIVO_TXT = r"C:\Users\leoed\OneDrive\Documentos\LamalYamine\base\Teste Notas.txt"



print(f"[DEBUG] Lendo arquivo bruto: {ARQUIVO_TXT}")
with open(ARQUIVO_TXT, encoding="utf-8") as f:
    conteudo_txt = f.read()


# Regex para separar cada Q&A (começa com número, espaço, hífen)
padrao = r"(?=\d+\s- )"
qna_chunks = re.split(padrao, conteudo_txt)
qna_chunks = [chunk.strip() for chunk in qna_chunks if chunk.strip()]

docs = []
for chunk in qna_chunks:
    # Tenta separar pergunta e resposta
    partes = chunk.split("\n", 1)
    pergunta = partes[0].strip() if len(partes) > 0 else ""
    resposta = partes[1].strip() if len(partes) > 1 else ""
    # Indexa Q&A completo
    docs.append(Document(page_content=chunk, metadata={"source": ARQUIVO_TXT, "tipo": "qna"}))
    # Indexa só a pergunta
    if pergunta:
        docs.append(Document(page_content=pergunta, metadata={"source": ARQUIVO_TXT, "tipo": "pergunta"}))
    # Indexa só a resposta
    if resposta:
        docs.append(Document(page_content=resposta, metadata={"source": ARQUIVO_TXT, "tipo": "resposta"}))
print(f"[DEBUG] Total de documentos (Q&A, perguntas e respostas): {len(docs)}")



# Não precisa mais de chunking, pois cada Q&A já é um documento
docs_split = docs
print(f"[DEBUG] Total de documentos para indexar: {len(docs_split)}")


print("[DEBUG] Criando embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
print("[DEBUG] Embeddings criados.")



print("[DEBUG] Persistindo DB...")
db = Chroma.from_documents(
    docs_split,
    embeddings,
    persist_directory=CAMINHO_DB
)
print("[DEBUG] DB persistido.")

print("Reindexação concluída! ✅")

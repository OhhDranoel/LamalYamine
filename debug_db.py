# debug_db.py
import os
from dotenv import load_dotenv

# 🔹 Carrega variáveis do .env
load_dotenv()

# 🔹 Sua chave da OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ A chave OPENAI_API_KEY não foi encontrada! Verifique seu arquivo .env")
    print("Exemplo de .env na raiz do projeto:")
    print("OPENAI_API_KEY=sua_chave_aqui")
    exit(1)

print("✅ Chave da OpenAI carregada corretamente!")

# 🔹 Importações do LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

# 🔹 Caminho do banco vetorial
CAMINHO_DB = "db"

# 🔹 Cria função de embeddings
func = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 🔹 Carrega o banco vetorial
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=func)

# 🔹 Pergunta de teste
pergunta = input("Digite uma pergunta para testar a base: ")

# 🔹 Busca similaridade (vários chunks)
resultados = db.similarity_search_with_relevance_scores(pergunta, k=10)

if not resultados:
    print("⚠️ Nenhum resultado encontrado na base de conhecimento.")
else:
    print(f"🔹 Foram encontrados {len(resultados)} chunks relevantes:\n")
    for i, (doc, score) in enumerate(resultados, start=1):
        src = doc.metadata.get("source") or doc.metadata.get("filename") or "sem fonte"
        print(f"#{i} Score: {score:.4f} | Fonte: {src}")
        print(doc.page_content[:400].replace("\n", " "))
        print("-" * 50)

# debug_db.py
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

CAMINHO_DB = "db"

func = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

db = Chroma(persist_directory=CAMINHO_DB, embedding_function=func)

# 🔹 Imprime todos os chunks do DB
print("\n🔹 TODOS os chunks indexados no banco:")
all_docs = db.get()
for i, doc in enumerate(all_docs["documents"], start=1):
    print(f"Chunk #{i} | Fonte: {all_docs['metadatas'][i-1].get('source', 'sem fonte')}")
    print(doc[:500].replace("\n", " "))  # mostra só os primeiros 500 caracteres
    print("-" * 80)

# 🔹 Pergunta de teste
pergunta = input("\nDigite uma pergunta para testar a base: ")

# 🔹 Busca similaridade (vários chunks)
resultados = db.similarity_search_with_relevance_scores(pergunta, k=10)

if not resultados:
    print("⚠️ Nenhum resultado encontrado na base de conhecimento.")
else:
    print(f"\n🔹 Resultados relevantes para a pergunta '{pergunta}':")
    for i, (doc, score) in enumerate(resultados, start=1):
        fonte = getattr(doc, "metadata", {}).get("source", "sem fonte")
        print(f"#{i} Score: {score:.4f} | Fonte: {fonte}")
        print(doc.page_content[:400].replace("\n", " "))
        print("-" * 50)

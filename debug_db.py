# debug_db.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

load_dotenv()
CAMINHO_DB = "db"

# 1️⃣ Conectar à base
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)

# 2️⃣ Listar todos os chunks
print("\n🔹 TODOS os chunks indexados no banco:")
all_docs = db.get()
for i, doc in enumerate(all_docs["documents"], start=1):
    print(f"Chunk #{i} | Fonte: {all_docs['metadatas'][i-1].get('source', 'sem fonte')}")
    print(doc[:1000].replace("\n", " "))  # mostra até 1000 caracteres
    print("-" * 80)

# 3️⃣ Pergunta de teste
pergunta = input("\nDigite uma pergunta para testar a base: ")

# 4️⃣ Buscar chunks relevantes
resultados = db.similarity_search_with_relevance_scores(pergunta, k=10)

if not resultados:
    print("⚠️ Nenhum resultado encontrado na base de conhecimento.")
else:
    print(f"\n🔹 Resultados relevantes para a pergunta '{pergunta}':")
    resposta_completa = ""
    for i, (doc, score) in enumerate(resultados, start=1):
        fonte = getattr(doc, "metadata", {}).get("source", "sem fonte")
        print(f"#{i} Score: {score:.4f} | Fonte: {fonte}")
        print(doc.page_content.replace("\n", " "))
        print("-" * 50)
        resposta_completa += doc.page_content + " "

    # Mostrar resposta completa unida
    print("\n🔹 Resposta completa unida:")
    print(resposta_completa)

# debug_db.py
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

CAMINHO_DB = "db"

func = OpenAIEmbeddings()
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=func)

q = "Quem descobriu o Brasil?"
res = db.similarity_search_with_relevance_scores(q, k=10)

print(f"Resultados encontrados: {len(res)}\n")
for i, (doc, score) in enumerate(res, start=1):
    src = doc.metadata.get("source") or doc.metadata.get("filename") or "sem fonte"
    print(f"#{i} score={score:.4f} source={src}")
    print(doc.page_content[:400].replace("\n", " ") + "\n" + "-"*40 + "\n")

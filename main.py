from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """
Responda a pergunta do usuário:
{pergunta}

com base nessas infomrações:

{base_conhecimento}

Se você não encontrar a resposta para a pergunta do usuário nessas informações,
responda não sei te dizer isso"""

pergunta = input("Escreva sua pergunta: ")

# carregar banco de dados
funcao_embedding = OpenAIEmbeddings()
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

# comparar a pergunta do usuário (embedding) com o meu banco de dados
resultados = db.similarity_search_with_relevance_scores
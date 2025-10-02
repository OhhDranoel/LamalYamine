# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Configurações
CAMINHO_DB = "db"
prompt_template = """
Responda a pergunta do usuário:
{pergunta}

com base nessas informações abaixo:

{base_conhecimento}
"""

# Inicializa Flask
app = Flask(__name__)
CORS(app)

# Carrega o banco Chroma ao iniciar o servidor
funcao_embedding = OpenAIEmbeddings()
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

def gerar_resposta(pergunta: str) -> str:
    # Busca por similaridade no banco
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=5)
    
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        return "Desculpe, não consegui encontrar informações relevantes na base de conhecimento."
    
    # Junta os textos relevantes
    textos_resultado = [resultado[0].page_content for resultado in resultados]
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    
    # Cria o prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_text = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})
    
    # Chama o modelo de linguagem
    modelo = ChatOpenAI()
    resposta = modelo.invoke(prompt_text).content
    return resposta

# Rota de teste
@app.route("/")
def home():
    return "Servidor rodando! 🚀"

# Rota de perguntas
@app.route("/perguntar", methods=["POST"])
def perguntar():
    data = request.json
    pergunta = data.get("pergunta", "")
    
    if not pergunta:
        return jsonify({"erro": "Nenhuma pergunta enviada"}), 400
    
    try:
        resposta = gerar_resposta(pergunta)
        return jsonify({"resposta": resposta})
    except Exception as e:
        return jsonify({"resposta": f"Erro ao processar a pergunta: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

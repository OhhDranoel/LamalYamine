# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 🔹 Carregar variáveis de ambiente (.env)
load_dotenv()

# Caminho do banco vetorial (criado com criar_db.py)
CAMINHO_DB = "db"

# 🔹 Inicializar Flask
app = Flask(__name__)
CORS(app)

# 🔹 Respostas fixas (não dependem dos documentos)
respostas_predefinidas = {
    "ola": "Olá! 👋 Como posso te ajudar hoje?",
    "olá": "Olá! 👋 Como posso te ajudar hoje?",
    "oi": "Oi! Tudo bem? 😊",
    "tudo bem?": "Estou funcionando perfeitamente! E você?",
    "pode me ajudar?": "Claro! É só perguntar que eu tento te ajudar com as informações da base. 🚀",
}

# 🔹 Template do prompt
prompt_template = """
Responda à pergunta do usuário:
{pergunta}

Com base nas informações abaixo (extraídas dos documentos):
{base_conhecimento}
"""

# Função principal para gerar resposta
def gerar_resposta(pergunta: str) -> str:
    # 1️⃣ Verificar respostas fixas
    pergunta_normalizada = pergunta.strip().lower()
    if pergunta_normalizada in respostas_predefinidas:
        return respostas_predefinidas[pergunta_normalizada]

    # 2️⃣ Carregar o banco vetorial
    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    # 3️⃣ Buscar trechos relevantes em TODOS os documentos
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=10)

    if len(resultados) == 0:
        return "Não consegui encontrar nenhuma informação relevante na base de conhecimento."

    # Filtrar apenas os trechos com relevância mínima
    textos_resultado = [
        resultado[0].page_content
        for resultado in resultados if resultado[1] >= 0.6
    ]

    if not textos_resultado:
        return "Encontrei informações, mas a relevância foi muito baixa."

    # 4️⃣ Concatenar os trechos em uma base de conhecimento
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)

    # 5️⃣ Montar prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_text = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    # 6️⃣ Chamar modelo da OpenAI
    modelo = ChatOpenAI()
    resposta = modelo.invoke(prompt_text).content
    return resposta


# 🔹 Rota de teste
@app.route('/')
def home():
    return "Servidor rodando! 🚀"


# 🔹 Rota para perguntas (usada pelo frontend)
@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.json
    pergunta = data.get('pergunta', '')

    if not pergunta:
        return jsonify({'erro': 'Nenhuma pergunta enviada'}), 400

    resposta = gerar_resposta(pergunta)
    return jsonify({'resposta': resposta})


if __name__ == '__main__':
    app.run(debug=True)
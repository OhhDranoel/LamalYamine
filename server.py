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
    "olá": "Olá! 👋 Como posso te ajudar hoje!",
    "oi": "Oi! Tudo bem? 😊",
    "tudo bem?": "Estou funcionando perfeitamente! E você?",
    "pode me ajudar?": "Claro! É só perguntar que eu tento te ajudar com as informações da base. 🚀",
    "obrigado": "De nada! Se precisar de mais alguma coisa, é só chamar. 😊",
    "obrigada": "De nada! Se precisar de mais alguma coisa, é só chamar. 😊",
    "valeu": "Por nada! Qualquer dúvida, estou aqui.",
    "agradecido": "Fico feliz em ajudar! Se precisar, é só perguntar.",
    "agradecida": "Fico feliz em ajudar! Se precisar, é só perguntar.",
    "bom dia": "Bom dia! Espero que seu dia seja ótimo! ☀️",
    "boa tarde": "Boa tarde! Como posso ajudar?",
    "boa noite": "Boa noite! Se precisar de algo, estou à disposição.",
}

# 🔹 Template do prompt
prompt_template = """
Responda à pergunta do usuário de forma clara, simpática e acolhedora, mas utilize apenas as informações fornecidas abaixo. Não adicione informações que não estejam presentes no texto. Não mencione que está usando documentos ou base de dados.

Pergunta:
{pergunta}

Informações fornecidas:
{base_conhecimento}

Se não souber a resposta, diga apenas 'Não sei.'
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



    # Sempre retorna pelo menos o chunk mais próximo
    if not resultados:
        return "Não consegui encontrar nenhuma informação relevante na base de conhecimento."

    # Tenta priorizar o chunk do tipo 'resposta'
    resposta_chunk = None
    for resultado in resultados:
        metadata = getattr(resultado[0], 'metadata', {})
        if metadata.get('tipo') == 'resposta':
            resposta_chunk = resultado[0].page_content
            break

    # Se não achar, pega o chunk mais próximo (primeiro resultado)
    if not resposta_chunk:
        resposta_chunk = resultados[0][0].page_content

    # 4️⃣ Montar base de conhecimento só com o chunk escolhido
    base_conhecimento = resposta_chunk

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
    try:
        data = request.json
        pergunta = data.get('pergunta', '')
        print(f"[LOG] Pergunta recebida: {pergunta}")

        if not pergunta:
            print("[LOG] Nenhuma pergunta enviada.")
            return jsonify({'erro': 'Nenhuma pergunta enviada'}), 400

        resposta = gerar_resposta(pergunta)
        print(f"[LOG] Resposta enviada: {resposta}")
        return jsonify({'resposta': resposta})
    except Exception as e:
        print(f"[ERRO] Erro ao processar a pergunta: {str(e)}")
        return jsonify({'erro': f'Erro ao processar a pergunta: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
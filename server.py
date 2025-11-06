# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import threading
import reindex

# 🔹 Carregar variáveis de ambiente (.env)
load_dotenv()

# Caminho do banco vetorial (criado com criar_db.py)
CAMINHO_DB = "db"

# 🔹 Inicializar Flask
app = Flask(__name__)
CORS(app)

# 🔹 Iniciar reindexação inicial e watcher em background
def _start_reindex_watcher():
    try:
        # Reindexa uma vez na inicialização para garantir que o DB exista/esteja atualizado
        print("[server] Executando reindexação inicial...")
        reindex.reindex()

        # Inicia thread daemon que observa alterações no arquivo de respostas
        watcher = threading.Thread(target=reindex.watch_file, args=(reindex.ARQUIVO_TXT, reindex.CAMINHO_DB, 2.0), daemon=True)
        watcher.start()
        print("[server] Watcher de reindexação iniciado.")
    except Exception as e:
        print(f"[server][ERRO] Não foi possível iniciar o watcher de reindex: {e}")


# Start watcher when module is imported/run
_start_reindex_watcher()

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

    # 6️⃣ Chamar modelo da OpenAI com timeout e tratamento de erros
    try:
        modelo = ChatOpenAI(timeout=30)  # Define timeout de 30 segundos
        resposta = modelo.invoke(prompt_text).content
        return resposta
    except Exception as e:
        print(f"[ERRO] Falha na comunicação com OpenAI: {str(e)}")
        return "Desculpe, estou tendo problemas de conexão no momento. Por favor, tente novamente em alguns instantes."


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

        # Aumentar o timeout da requisição para 60 segundos
        resposta = gerar_resposta(pergunta)
        print(f"[LOG] Resposta enviada: {resposta}")
        return jsonify({
            'status': 'sucesso',
            'resposta': resposta
        })
    except Exception as e:
        erro_msg = str(e)
        print(f"[ERRO] Erro ao processar a pergunta: {erro_msg}")
        
        if "connect" in erro_msg.lower():
            mensagem = "Problema de conexão com o servidor. Por favor, verifique sua internet e tente novamente."
        else:
            mensagem = "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente em alguns instantes."
            
        return jsonify({
            'status': 'erro',
            'mensagem': mensagem,
            'erro': erro_msg
        }), 500


@app.route('/reindex', methods=['POST'])
def trigger_reindex():
    """Rota para forçar reindexação manual (útil para depuração)."""
    try:
        print("[server] Requisição HTTP: reindex -> iniciando reindexação...")
        ok = reindex.reindex()
        if ok:
            return jsonify({'status': 'ok', 'mensagem': 'Reindexação concluída.'})
        else:
            return jsonify({'status': 'error', 'mensagem': 'Falha ao reindexar. Veja logs no servidor.'}), 500
    except Exception as e:
        print(f"[server][ERRO] Exceção em /reindex: {e}")
        return jsonify({'status': 'error', 'mensagem': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS

# Cria o app Flask
app = Flask(__name__)
CORS(app)  # Permite requisições do navegador (CORS)

# Banco de respostas de exemplo (simula um modelo ou LangChain)
respostas_exemplo = {
    "qual a grade curricular?": "A grade curricular inclui Matemática, Português, História, Ciências e Educação Física.",
    "qual o horário das aulas?": "As aulas começam às 7h e terminam às 17h.",
    "quem é o diretor?": "O diretor da escola é o Sr. João Silva.",
    "onde fica a escola?": "A escola fica na Rua das Flores, 123, Cidade XYZ."
}

def gerar_resposta(pergunta):
    """
    Simula a lógica de um modelo de linguagem ou LangChain.
    Retorna a resposta se conhecida, ou mensagem padrão se não.
    """
    pergunta_normalizada = pergunta.strip().lower()
    return respostas_exemplo.get(pergunta_normalizada, "Desculpe, não sei a resposta para isso.")

# Rota de teste
@app.route('/')
def home():
    return "Servidor rodando! 🚀"

# Rota para receber perguntas do HTML
@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.json  # Recebe JSON do frontend
    pergunta = data.get('pergunta', '')

    if not pergunta:
        return jsonify({'erro': 'Nenhuma pergunta enviada'}), 400

    resposta = gerar_resposta(pergunta)

    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    # Rodando o servidor em debug (reinicia automático quando muda código)
    app.run(debug=True)

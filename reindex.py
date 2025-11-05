"""reindex.py

Funções utilitárias para (re)indexar o arquivo de respostas.
Este módulo agora expõe `reindex()` e `watch_file()` para uso pelo servidor.
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

load_dotenv()

# Configurações
CAMINHO_DB = "db"
ARQUIVO_TXT = r"C:\Users\leoed\OneDrive\Documentos\LamalYamine-1\base\Teste Notas.txt"


def _read_respostas_from_file(path: str):
    """Lê o arquivo e retorna uma lista de respostas (cada bloco separado por linha em branco dupla)."""
    print(f"[reindex] Lendo arquivo: {path}")
    with open(path, encoding="utf-8") as f:
        conteudo_txt = f.read()

    respostas = [resp.strip() for resp in conteudo_txt.split("\n\n") if resp.strip()]
    return respostas


def reindex(caminho_txt: str = ARQUIVO_TXT, caminho_db: str = CAMINHO_DB):
    """Reindexa o conteúdo do arquivo plain-text no Chroma DB (substitui os dados persistidos).

    Essa função sobrescreve o diretório de persistência do Chroma com os novos embeddings.
    """
    try:
        respostas = _read_respostas_from_file(caminho_txt)
        docs = []
        for resposta in respostas:
            docs.append(Document(page_content=resposta, metadata={"source": caminho_txt, "tipo": "resposta"}))

        print(f"[reindex] Total de documentos: {len(docs)}")

        print("[reindex] Criando embeddings...")
        embeddings = OpenAIEmbeddings()
        print("[reindex] Embeddings criados.")

        print("[reindex] Persistindo DB...")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=caminho_db
        )
        print("[reindex] DB persistido.")
        print("[reindex] Reindexação concluída! ✅")
        return True
    except Exception as e:
        print(f"[reindex][ERRO] Falha ao reindexar: {e}")
        return False


def watch_file(path: str = ARQUIVO_TXT, caminho_db: str = CAMINHO_DB, interval: float = 2.0):
    """Observa o arquivo `path` e chama `reindex()` sempre que ele for modificado.

    Usa polling simples (mtime). Roda indefinidamente até o processo ser finalizado.
    """
    try:
        last_mtime = os.path.getmtime(path) if os.path.exists(path) else None
    except Exception:
        last_mtime = None

    print(f"[reindex/watch] Observando {path} (interval={interval}s)")
    while True:
        try:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if last_mtime is None:
                    # primeira vez
                    print("[reindex/watch] Arquivo encontrado. Executando reindex pela primeira vez.")
                    reindex(path, caminho_db)
                    last_mtime = mtime
                elif mtime != last_mtime:
                    print(f"[reindex/watch] Mudança detectada (mtime {mtime}). Reindexando...")
                    reindex(path, caminho_db)
                    last_mtime = mtime
        except Exception as e:
            print(f"[reindex/watch][ERRO] {e}")

        time.sleep(interval)


if __name__ == '__main__':
    # Permite executar o script manualmente: 'py reindex.py' fará uma reindexação única.
    reindex()
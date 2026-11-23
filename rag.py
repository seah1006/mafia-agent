# =====================================
# rag.py : 룰북 PDF -> 메타데이터 -> 청킹 -> Chroma 벡터DB -> 필터 검색
# =====================================
import gc
import shutil
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config

COLLECTION = "mafia_rulebook"


# =====================================
# 1. PDF 로드 + 페이지 메타데이터 부착
# =====================================
def load_pdf_with_meta(pdf_path=config.PDF_PATH):
    """PDF 를 페이지 단위로 읽고 config.PAGE_META 의 type/role/name/title 을 붙인다.
    각 페이지 첫 줄이 약속된 제목과 다르면 한국어 오류를 낸다."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError("룰북 PDF 를 찾을 수 없습니다: " + str(pdf_path))
    docs = PyPDFLoader(str(pdf_path)).load()
    if len(docs) != len(config.PAGE_META):
        raise ValueError(
            "PDF 페이지 수가 맞지 않습니다: 기대 %d쪽, 실제 %d쪽"
            % (len(config.PAGE_META), len(docs))
        )
    for doc in docs:
        page = doc.metadata.get("page")
        meta = config.PAGE_META[page]
        lines = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()]
        first = lines[0] if lines else ""
        if meta["title"] not in first:
            raise ValueError(
                "PDF %d쪽(0부터)의 첫 줄이 제목과 다릅니다. 기대: '%s' / 실제: '%s'"
                % (page, meta["title"], first)
            )
        doc.metadata.update(
            {"type": meta["type"], "role": meta["role"], "name": meta["name"], "title": meta["title"]}
        )
    return docs


# =====================================
# 2. 청킹
# =====================================
def split_docs(docs, chunk_size=500, chunk_overlap=50):
    """RecursiveCharacterTextSplitter 로 분할. 같은 페이지 안 순서를 chunk_idx 로 남긴다."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    counter = {}
    for ch in chunks:
        key = ch.metadata.get("page")
        ch.metadata["chunk_idx"] = counter.get(key, 0)
        counter[key] = counter.get(key, 0) + 1
    return chunks


# =====================================
# 3. 벡터DB 생성 / 로드
# =====================================
def _embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def _release_chroma():
    """Windows 파일 잠금 방지: Chroma 클라이언트 캐시를 정리한다."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


def _reset_db(db_dir):
    db_dir = Path(db_dir)
    if not db_dir.exists():
        return
    # 먼저 컬렉션을 비운 뒤(잠금과 무관), 폴더 삭제는 클라이언트를 닫고 재시도한다
    try:
        old = Chroma(collection_name=COLLECTION, persist_directory=str(db_dir),
                     embedding_function=_embeddings())
        old.delete_collection()
        del old
    except Exception:
        pass
    _release_chroma()
    for _ in range(5):
        try:
            shutil.rmtree(db_dir)
            return
        except OSError:
            time.sleep(0.5)
            _release_chroma()
    # 폴더를 못 지워도 컬렉션은 이미 비어 있으므로 계속 진행한다


def build_db(pdf_path=config.PDF_PATH, db_dir=config.DB_DIR, reset=True):
    """PDF -> 메타데이터 -> 청킹 -> 임베딩 -> Chroma(영구 저장)"""
    if reset:
        _reset_db(db_dir)
    chunks = split_docs(load_pdf_with_meta(pdf_path))
    vs = Chroma(collection_name=COLLECTION, persist_directory=str(db_dir),
                embedding_function=_embeddings())
    if not reset and vs.get()["ids"]:
        vs.delete_collection()
        vs = Chroma(collection_name=COLLECTION, persist_directory=str(db_dir),
                    embedding_function=_embeddings())
    vs.add_documents(chunks)
    return vs


def load_db(db_dir=config.DB_DIR):
    vs = Chroma(collection_name=COLLECTION, persist_directory=str(db_dir),
                embedding_function=_embeddings())
    if not vs.get(limit=1)["ids"]:
        raise RuntimeError("벡터DB 가 비어 있습니다. 먼저 rag.build_db() 를 실행하세요.")
    return vs


# =====================================
# 4. 메타데이터 필터 검색
# =====================================
def search_rules(vs, query, k=3):
    return vs.similarity_search(query, k=k, filter={"type": "rule"})


def search_strategy(vs, role, query, k=2):
    flt = {"$and": [{"type": "strategy"}, {"role": role}]}
    return vs.similarity_search(query, k=k, filter=flt)


def get_persona(vs, name):
    """캐릭터 프로필 페이지를 청크 순서대로 이어 붙여 돌려준다(유사도 검색 불필요)."""
    res = vs.get(where={"$and": [{"type": "character"}, {"name": name}]})
    items = sorted(zip(res["documents"], res["metadatas"]), key=lambda x: x[1].get("chunk_idx", 0))
    return "\n".join(d for d, _ in items)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 .env에 없습니다"

PDF_PATH = "data/pdf/InternetServiceToU.pdf"
FAISS_INDEX_PATH = "faiss_index"


def is_amendment_page(text: str) -> bool:
    """부칙/시행일 위주 페이지 판별 (개정 이력 페이지 제거용)"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    amendment_keywords = ("부    칙", "부칙", "시행일", "경과조치")
    keyword_lines = sum(1 for l in lines if any(k in l for k in amendment_keywords))
    return keyword_lines / len(lines) > 0.4


def main():
    print("PDF 로드 중...")
    loader = PyMuPDFLoader(PDF_PATH)
    raw_docs = loader.load()
    print(f"전체 페이지: {len(raw_docs)}")

    filtered_docs = [doc for doc in raw_docs if not is_amendment_page(doc.page_content)]
    print(f"필터링 후 페이지: {len(filtered_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(filtered_docs)
    print(f"총 청크: {len(chunks)}")

    print("임베딩 생성 중... (OpenAI API 호출)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS 인덱스 저장 완료: {FAISS_INDEX_PATH}/")


if __name__ == "__main__":
    main()

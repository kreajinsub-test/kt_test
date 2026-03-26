from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_graph_retriever.transformers import ShreddingTransformer
from graph_rag_example_helpers.datasets.animals import fetch_documents
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "text-embedding-3-small")

print(f"API KEY : {OPENAI_API_KEY[:10]}...")
print(f"LLM     : {MODEL_NAME}")
print(f"EMB     : {EMB_MODEL_NAME}")

# ── 모델 초기화 ────────────────────────────────────────────
emb = OpenAIEmbeddings(model=EMB_MODEL_NAME, api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

# ── 데이터 로드 & 전처리 ───────────────────────────────────
print("\n[1] 데이터 로드 중...")
raw_docs = fetch_documents()
print(f"  raw_docs: {len(raw_docs)}개")

docs = ShreddingTransformer().transform_documents(raw_docs)
animal_docs = filter_complex_metadata(docs)
print(f"  animal_docs: {len(animal_docs)}개")

# ── 벡터 스토어 구축 ───────────────────────────────────────
print("\n[2] 벡터 스토어 구축 중...")
vector_store = InMemoryVectorStore.from_documents(
    documents=animal_docs,
    embedding=emb,
)

# ── 일반 벡터 검색 비교 ────────────────────────────────────
print("\n[3] 일반 벡터 검색 결과:")
query = "What animals can be found near a capybara?"
for doc in vector_store.similarity_search(query, k=5):
    print(f"  - {doc.id}: {doc.page_content[:80]}")

# ── Graph Retriever ────────────────────────────────────────
print("\n[4] Graph Retriever 구축 중...")
traversal_retriever = GraphRetriever(
    store=vector_store,
    edges=[
        ("type", "type"),
        ("habitat", "habitat"),
        ("number_of_legs", "number_of_legs"),
        ("diet", "diet"),
        ("origin", "origin"),
    ],
    strategy=Eager(k=8, start_k=15, max_depth=2),
)

print("\n[5] Graph Retriever 검색 결과:")
for doc in traversal_retriever.invoke(query):
    print(f"  - {doc.id}: {doc.page_content[:80]}")

# ── standard_retriever (실험용 파라미터) ───────────────────
standard_retriever = GraphRetriever(
    store=vector_store,
    edges=[("habitat", "habitat"), ("origin", "origin")],
    strategy=Eager(k=10, start_k=15, max_depth=0),
)

# ── RAG Chain (번역 포함) ──────────────────────────────────
print("\n[6] RAG Chain 구성 중...")

translate_for_retrieval = (
    ChatPromptTemplate.from_template(
        """사용자의 질문을 RAG를 위한 검색용으로 적합한 간결한 영어 키워드로 번역하세요.
고유명사는 그대로 유지하세요.
번역된 질의문만 출력하세요.

질문: {question}"""
    )
    | llm
    | StrOutputParser()
)


def format_docs(docs):
    return "\n\n".join(
        f"text: {d.page_content}\nmetadata: {d.metadata}" for d in docs
    )


chain = (
    {
        "question": RunnablePassthrough(),
        "context": translate_for_retrieval | standard_retriever | format_docs,
    }
    | ChatPromptTemplate.from_template(
        """아래 컨텍스트만 근거로 문장으로 답하세요.
        답변을 하고 나서 근거를 알기 위해 컨택스트도 정리해서 마지막에 예쁘게 불릿리스트로 출력해주세요.

##컨텍스트:{context}

##질문: {question}"""
    )
    | llm
    | StrOutputParser()
)

# ── 테스트 ─────────────────────────────────────────────────
ko_query = "카피바라 근처에서 발견할 수 있는 동물은 뭐야?"
print(f"\n[7] 질문: {ko_query}")

translated = translate_for_retrieval.invoke(ko_query)
print(f"  번역된 질문: {translated}")

answer = chain.invoke(ko_query)
print(f"\n  답변: {answer}")

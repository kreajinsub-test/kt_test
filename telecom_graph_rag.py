"""
통신사 정보 Graph RAG + 웹 검색 통합
- 그래프 DB는 최초 1회만 생성하고 이후에는 pickle에서 로드합니다.
- 그래프를 다시 빌드하려면: python telecom_graph_rag.py --rebuild
- 각 질문에 대해 그래프 검색과 인터넷 검색을 병렬로 수행한 뒤 통합 답변을 생성합니다.
"""

import asyncio
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

GRAPH_CACHE_PATH = "telecom_graph.pkl"

TITLES = [
    "KT Corporation",
    "AT&T",
    "Avaya",
    "Qwest",
    "Sprint Corporation",
    "China Mobile",
    "China Telecom",
    "KDDI",
    "Nippon Telegraph and Telephone",
]

BATCH_SIZE = 16
MAX_CONCURRENCY = 4


# ── 문서 로드 ──────────────────────────────────────────────
def load_docs(titles: list[str]) -> list[Document]:
    print(f"  Wikipedia API로 {len(titles)}개 문서 로드 중...")
    docs = []
    for title in titles:
        try:
            loaded = WikipediaLoader(query=title, load_max_docs=1, lang="en").load()
            docs.extend(loaded)
            print(f"    ✓ {title}: {len(loaded[0].page_content)}자")
        except Exception as e:
            print(f"    ✗ {title}: {e}")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    print(f"  문서 {len(docs)}개 → 청크 {len(splits)}개")
    return splits


# ── 그래프 빌드 ────────────────────────────────────────────
async def build_graph_async(splits: list[Document], llm) -> NetworkxEntityGraph:
    transformer = LLMGraphTransformer(llm=llm)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    def batched(it, n):
        it = iter(it)
        while chunk := list(islice(it, n)):
            yield chunk

    async def worker(batch):
        async with sem:
            return await transformer.aconvert_to_graph_documents(batch)

    batches = list(batched(splits, BATCH_SIZE))
    tasks = [asyncio.create_task(worker(b)) for b in batches]

    all_graph_docs = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(batches), desc="  그래프 변환"):
        gds = await coro
        all_graph_docs.extend(gds)

    graph = NetworkxEntityGraph()
    for gd in all_graph_docs:
        for n in gd.nodes:
            graph.add_node(str(getattr(n, "id", n)))
        for r in gd.relationships:
            src = str(getattr(r.source, "id", r.source))
            tgt = str(getattr(r.target, "id", r.target))
            rel = getattr(r, "type", "RELATED_TO")
            graph._graph.add_edge(src, tgt, relation=rel)

    graph._graph = graph._graph.to_undirected()
    print(f"  노드: {len(graph._graph.nodes)}개, 엣지: {len(graph._graph.edges)}개")
    return graph


def get_or_build_graph(llm, rebuild: bool = False) -> NetworkxEntityGraph:
    if not rebuild and os.path.exists(GRAPH_CACHE_PATH):
        print(f"[DB] 캐시 로드: {GRAPH_CACHE_PATH}")
        with open(GRAPH_CACHE_PATH, "rb") as f:
            nx_graph = pickle.load(f)
        graph = NetworkxEntityGraph()
        graph._graph = nx_graph
        print(f"  노드: {len(graph._graph.nodes)}개, 엣지: {len(graph._graph.edges)}개")
        return graph

    print("[DB] 그래프 신규 빌드 중...")
    splits = load_docs(TITLES)
    graph = asyncio.run(build_graph_async(splits, llm))

    with open(GRAPH_CACHE_PATH, "wb") as f:
        pickle.dump(graph._graph, f)
    print(f"[DB] 그래프 저장 완료: {GRAPH_CACHE_PATH}")
    return graph


# ── 그래프 컨텍스트 조회 ────────────────────────────────────
def get_graph_context(
    query: str,
    entity_chain,
    graph: NetworkxEntityGraph,
) -> tuple[str, list[str]]:
    """엔티티를 추출하고 그래프에서 관련 트리플을 조회합니다."""
    entities_str = entity_chain.invoke({"input": query})
    entities = [e.strip() for e in entities_str.split(",") if e.strip()]

    # 대소문자 무시 매칭
    node_map = {n.lower(): n for n in graph._graph.nodes}
    matched = [node_map[e.lower()] for e in entities if e.lower() in node_map]

    triples = []
    for entity in matched:
        triples.extend(graph.get_entity_knowledge(entity))

    context = "\n".join(triples) if triples else "(그래프에서 관련 정보 없음)"
    return context, triples


# ── 웹 검색 ────────────────────────────────────────────────
def _sanitize(text: str) -> str:
    """제어문자 및 이상한 유니코드를 제거합니다."""
    return "".join(c for c in text if c.isprintable() or c in ("\n", "\t"))


def get_web_context(
    query: str,
    web_search: DuckDuckGoSearchAPIWrapper,
    num_results: int = 5,
) -> tuple[str, list[dict]]:
    """DuckDuckGo로 웹 검색하고 결과를 반환합니다."""
    try:
        results = web_search.results(query, num_results)
    except Exception as e:
        print(f"  [웹검색 오류] {e}")
        return "(웹 검색 실패)", []

    snippets = [
        f"[{_sanitize(r['title'])}]\n{_sanitize(r['snippet'])}" for r in results
    ]
    context = "\n\n".join(snippets)
    return context, results


# ── 통합 질의 ──────────────────────────────────────────────
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """당신은 통신사 정보 전문 QA 어시스턴트입니다.
아래 두 출처의 정보를 모두 활용해 한국어로 답하세요.
정보가 부족한 부분은 부족하다고 말하세요.

## 그래프 DB (지식 트리플)
{graph_context}

## 웹 검색 결과
{web_context}

## 질문
{question}

답변:"""
)

TRANSLATE_PROMPT = ChatPromptTemplate.from_template(
    """다음 질문을 통신사 관련 웹 검색에 적합한 간결한 영어 쿼리로 번역하세요.
번역된 쿼리만 출력하세요.

질문: {question}
영어 쿼리:"""
)


def ask(
    query: str,
    entity_chain,
    graph: NetworkxEntityGraph,
    web_search: DuckDuckGoSearchAPIWrapper,
    llm,
) -> str:
    # 웹 검색용 영어 번역 (한글 쿼리는 관련 없는 결과가 나올 수 있음)
    en_query = (TRANSLATE_PROMPT | llm | StrOutputParser()).invoke({"question": query})

    # 그래프 검색 + 웹 검색 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_graph = executor.submit(get_graph_context, query, entity_chain, graph)
        future_web = executor.submit(get_web_context, en_query, web_search)
        graph_context, triples = future_graph.result()
        web_context, web_results = future_web.result()

    # LLM 답변 생성
    answer = (ANSWER_PROMPT | llm | StrOutputParser()).invoke({
        "graph_context": graph_context,
        "web_context": web_context,
        "question": query,
    })

    # 레퍼런스 섹션 (실제 데이터 기반, LLM 생성 아님)
    graph_refs = "\n".join(f"  - {t}" for t in triples) or "  (없음)"
    web_refs = "\n".join(
        f"  - {r['title']}\n    {r['link']}" for r in web_results
    ) or "  (없음)"

    return (
        f"{answer}\n\n"
        f"── 근거 레퍼런스 ────────────────────────────\n"
        f"[그래프 트리플]\n{graph_refs}\n\n"
        f"[웹 출처]\n{web_refs}"
    )


# ── 메인 ──────────────────────────────────────────────────
def main():
    rebuild = "--rebuild" in sys.argv

    print(f"LLM: {MODEL_NAME}")
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
    web_search = DuckDuckGoSearchAPIWrapper()

    graph = get_or_build_graph(llm, rebuild=rebuild)

    # 엔티티 추출 체인
    candidate_entities = list(graph._graph.nodes)
    entity_chain = (
        PromptTemplate(
            input_variables=["input", "entities"],
            template="""다음 질문에 답하려면 아래 '그래프 엔티티 후보' 중에서 관련된 엔티티 이름을 1~5개 고르세요.
- 정확히 '엔티티 이름'만 쉼표로 출력하세요.
- 후보에 없으면 고르지 마세요.

[그래프 엔티티 후보]
{entities}

[질문]
{input}

[정답](쉼표로 구분된 엔티티들만):""",
        ).partial(entities="\n".join(map(str, candidate_entities)))
        | llm
        | StrOutputParser()
    )

    queries = [
        "Kt의 창립일은 언제야?",
        "Kt의 주요 사업은 뭐야?",
        "미국의 통신사는 어떤 곳이 있어?",
        "일본의 두 통신사를 비교해줘",
    ]

    print("\n" + "=" * 60)
    for i, query in enumerate(queries, 1):
        print(f"\n[Q{i}] {query}")
        result = ask(query, entity_chain, graph, web_search, llm)
        print(result)
        print("=" * 60)


if __name__ == "__main__":
    main()

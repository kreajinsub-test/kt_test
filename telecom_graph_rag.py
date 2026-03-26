"""
통신사 정보 Graph RAG
- 그래프 DB는 최초 1회만 생성하고 이후에는 pickle에서 로드합니다.
- 그래프를 다시 빌드하려면: python telecom_graph_rag.py --rebuild
"""

import asyncio
import os
import pickle
import sys
from itertools import islice

from dotenv import load_dotenv
from langchain_community.chains.graph_qa.base import GraphQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
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

    # NetworkxEntityGraph 구성
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
    print(
        f"  노드: {len(graph._graph.nodes)}개, 엣지: {len(graph._graph.edges)}개"
    )
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


# ── GraphQAChain 구성 ──────────────────────────────────────
def build_chain(llm, graph: NetworkxEntityGraph) -> GraphQAChain:
    candidate_entities = list(graph._graph.nodes)

    entity_prompt = PromptTemplate(
        input_variables=["input", "entities"],
        template="""다음 질문에 답하려면 아래 '그래프 엔티티 후보' 중에서 관련된 엔티티 이름을 1~5개 고르세요.
- 정확히 '엔티티 이름'만 쉼표로 출력하세요.
- 후보에 없으면 고르지 마세요.

[그래프 엔티티 후보]
{entities}

[질문]
{input}

[정답](쉼표로 구분된 엔티티들만):""",
    )

    qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""당신은 지식그래프 QA 어시스턴트입니다.
주어진 엔티티 주변 트리플을 근거로 한국어로 간결하게 답하세요.
관련 근거의 레퍼런스를 트리플 형태로 같이 제시하세요.
그래프와 컨텍스트에서 찾을 수 없다면 관련 근거가 없다고 말하세요.

형식 : [node1] → [edge] → [node2]

[질문]
{question}

[컨텍스트]
{context}

[답변]""",
    )

    return GraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        entity_prompt=entity_prompt.partial(
            entities="\n".join(map(str, candidate_entities))
        ),
        qa_prompt=qa_prompt,
        verbose=False,
    )


# ── 메인 ──────────────────────────────────────────────────
def main():
    rebuild = "--rebuild" in sys.argv

    print(f"LLM: {MODEL_NAME}")
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

    graph = get_or_build_graph(llm, rebuild=rebuild)
    chain = build_chain(llm, graph)

    queries = [
        "Kt의 창립일은 언제야?",
        "Kt의 주요 사업은 뭐야?",
        "Kt는 어디에 위치해 있어?",
        "미국의 통신사는 어떤 곳이 있어?",
        "일본의 두 통신사를 비교해줘",
    ]

    print("\n" + "=" * 60)
    for i, query in enumerate(queries, 1):
        print(f"\n[Q{i}] {query}")
        result = chain.invoke({"query": query})["result"]
        print(f"[A{i}] {result}")
        print("-" * 60)


if __name__ == "__main__":
    main()

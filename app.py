import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
import gradio as gr
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 .env에 없습니다"

FAISS_INDEX_PATH = "faiss_index"

# ── 공통: 벡터 스토어 / LLM 로드 ─────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError("faiss_index가 없습니다. 먼저 embed.py를 실행하세요.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ══════════════════════════════════════════════════════════════════
# 1. 일반 RAG
# ══════════════════════════════════════════════════════════════════
def build_rag_chain():
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "대화 기록과 최신 질문을 보고, 대화 기록 없이도 이해할 수 있는 독립적인 검색 쿼리로 재작성하세요. 질문에 답하지 말고, 필요하면 재작성하고 그렇지 않으면 그대로 반환하세요."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    def retrieve_with_history(x: dict) -> str:
        query = contextualize_chain.invoke(x) if x["chat_history"] else x["input"]
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 KT 인터넷 이용약관 문서를 기반으로 답변하는 어시스턴트입니다.
아래 컨텍스트를 바탕으로 질문에 최대한 성실하게 답하세요.
컨텍스트에 명확한 정의가 없더라도, 관련 정보가 있다면 그것을 바탕으로 답변하세요.
컨텍스트에 전혀 관련 내용이 없을 때만 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

[컨텍스트]
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    return (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_with_history))
        | qa_prompt
        | llm
        | StrOutputParser()
    )


rag_chain = build_rag_chain()


def rag_respond(message: str, history: list):
    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return rag_chain.invoke({"input": message, "chat_history": chat_history})


# ══════════════════════════════════════════════════════════════════
# 2. Agentic RAG
# ══════════════════════════════════════════════════════════════════
agent_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


@tool
def search_doc(query: str) -> str:
    """KT 인터넷 이용약관 문서에서 관련 내용을 검색합니다.
    가격, 약관 조항, 서비스 정책 등을 찾을 때 사용하세요.
    검색어를 명확하고 구체적으로 작성할수록 좋은 결과가 나옵니다."""
    docs = agent_retriever.invoke(query)
    if not docs:
        return "관련 문서를 찾지 못했습니다."
    return "\n\n".join(f"[문서 {i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1))


agent_system_prompt = """당신은 KT 인터넷 이용약관 전문 어시스턴트입니다.
사용자의 질문에 답하기 위해 search_doc 도구를 적극 활용하세요.

행동 지침:
- 질문이 복잡하거나 여러 주제를 포함하면 검색을 여러 번 나눠서 수행하세요.
- 첫 검색 결과가 불충분하면 다른 키워드로 재검색하세요.
- 문서에서 찾은 내용만을 근거로 답변하고, 출처가 불분명한 내용은 추측임을 명시하세요.
- 가격 정보는 정확하게 인용하세요."""

agent_memory = MemorySaver()
agent = create_react_agent(
    llm,
    tools=[search_doc],
    prompt=agent_system_prompt,
    checkpointer=agent_memory,
)


def agent_respond(message: str, history: list, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


# ══════════════════════════════════════════════════════════════════
# 3. Advanced RAG (Corrective RAG with LangGraph)
# ══════════════════════════════════════════════════════════════════

# ── Pydantic 평가 스키마 ──────────────────────────────────────────
class GradeDoc(BaseModel):
    score: Literal["yes", "no"] = Field(description="문서가 질문과 관련 있으면 'yes', 없으면 'no'")

class GradeHallucination(BaseModel):
    score: Literal["yes", "no"] = Field(description="답변이 문서에 근거하면 'yes', 근거 없이 지어낸 내용이면 'no'")

class GradeAnswer(BaseModel):
    score: Literal["yes", "no"] = Field(description="답변이 질문을 충분히 해결하면 'yes', 아니면 'no'")


# ── Grader / Generator 체인 ──────────────────────────────────────
doc_grader = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 검색된 문서가 질문과 관련 있는지 평가하는 채점자입니다. 키워드나 의미가 관련 있으면 'yes', 관련 없으면 'no'로만 답하세요."),
        ("human", "질문: {question}\n\n문서:\n{document}"),
    ])
    | llm.with_structured_output(GradeDoc)
)

hallucination_grader = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 LLM 답변이 제공된 문서에 근거하는지 평가합니다. 문서에 근거하면 'yes', 문서에 없는 내용을 지어냈으면 'no'로만 답하세요."),
        ("human", "문서:\n{documents}\n\n답변: {generation}"),
    ])
    | llm.with_structured_output(GradeHallucination)
)

answer_grader = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 답변이 질문을 충분히 해결하는지 평가합니다. 해결하면 'yes', 아니면 'no'로만 답하세요."),
        ("human", "질문: {question}\n\n답변: {generation}"),
    ])
    | llm.with_structured_output(GradeAnswer)
)

query_rewriter = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 검색 쿼리를 더 나은 버전으로 재작성합니다. 질문의 핵심 의도를 파악하고 검색에 최적화된 쿼리로 바꿔주세요. 재작성된 쿼리만 출력하세요."),
        ("human", "원래 질문: {question}"),
    ])
    | llm
    | StrOutputParser()
)

advanced_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 KT 인터넷 이용약관 문서를 기반으로 답변하는 어시스턴트입니다.
아래 컨텍스트를 바탕으로 질문에 최대한 성실하게 답하세요.
컨텍스트에 명확한 정의가 없더라도, 관련 정보가 있다면 그것을 바탕으로 답변하세요.
컨텍스트에 전혀 관련 내용이 없을 때만 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

[컨텍스트]
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

advanced_generator = advanced_qa_prompt | llm | StrOutputParser()

advanced_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

MAX_RETRIES = 2


# ── Graph State ───────────────────────────────────────────────────
class CRAGState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int
    chat_history: List


# ── Graph Nodes ───────────────────────────────────────────────────
def retrieve(state: CRAGState) -> dict:
    docs = advanced_retriever.invoke(state["question"])
    return {"documents": docs}


def grade_documents(state: CRAGState) -> dict:
    relevant = []
    for doc in state["documents"]:
        result = doc_grader.invoke({"question": state["question"], "document": doc.page_content})
        if result.score == "yes":
            relevant.append(doc)
    return {"documents": relevant}


def generate(state: CRAGState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    generation = advanced_generator.invoke({
        "context": context,
        "question": state["question"],
        "chat_history": state["chat_history"],
    })
    return {"generation": generation}


def rewrite_query(state: CRAGState) -> dict:
    new_question = query_rewriter.invoke({"question": state["question"]})
    return {"question": new_question, "retries": state["retries"] + 1}


# ── Conditional Edges ─────────────────────────────────────────────
def route_after_grading(state: CRAGState) -> Literal["generate", "rewrite_query"]:
    if state["documents"] or state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite_query"


def route_after_generation(state: CRAGState) -> Literal["end", "rewrite_query", "generate"]:
    docs_text = "\n\n".join(doc.page_content for doc in state["documents"])
    hallucination = hallucination_grader.invoke({
        "documents": docs_text,
        "generation": state["generation"],
    })
    if hallucination.score == "no":
        # 환각 발생 → 재생성 (쿼리 재작성 없이)
        return "generate" if state["retries"] < MAX_RETRIES else "end"

    answer = answer_grader.invoke({
        "question": state["question"],
        "generation": state["generation"],
    })
    if answer.score == "yes":
        return "end"
    # 답변 불충분 → 쿼리 재작성
    return "rewrite_query" if state["retries"] < MAX_RETRIES else "end"


# ── Graph 구성 ────────────────────────────────────────────────────
def build_crag_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate", generate)
    g.add_node("rewrite_query", rewrite_query)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading)
    g.add_conditional_edges("generate", route_after_generation, {
        "end": END,
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    })
    g.add_edge("rewrite_query", "retrieve")

    return g.compile()


crag_graph = build_crag_graph()


def advanced_respond(message: str, history: list):
    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    result = crag_graph.invoke({
        "question": message,
        "documents": [],
        "generation": "",
        "retries": 0,
        "chat_history": chat_history,
    })
    return result["generation"]


# ══════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(title="KT 인터넷 이용약관 챗봇") as demo:
    gr.Markdown("# KT 인터넷 이용약관 챗봇")

    with gr.Tabs():
        with gr.Tab("일반 RAG"):
            gr.Markdown("질문 → 1회 검색 → 답변. 빠르고 간단한 질문에 적합합니다.")
            gr.ChatInterface(
                fn=rag_respond,
                examples=["기가 와이파이는 뭐야?", "서비스 해지 시 환불 정책은?", "이용정지 사유는 무엇인가요?"],
            )

        with gr.Tab("Agentic RAG"):
            gr.Markdown("에이전트가 스스로 검색 전략을 결정합니다. 복잡하거나 다단계 질문에 적합합니다.")
            session_id = gr.State(value="default-session")
            gr.ChatInterface(
                fn=agent_respond,
                additional_inputs=[session_id],
                examples=[
                    ["기가 와이파이 상품의 종류와 가격을 모두 알려줘", "default-session"],
                    ["서비스 이용정지 사유와 해지 절차를 비교해서 설명해줘", "default-session"],
                    ["약정 기간별로 위약금은 어떻게 달라져?", "default-session"],
                ],
            )

        with gr.Tab("Advanced RAG (CRAG)"):
            gr.Markdown("""**Corrective RAG**: 검색 → 문서 관련성 평가 → 쿼리 재작성(필요시) → 생성 → 환각 검사
- 관련 없는 문서는 자동 필터링
- 답변 품질 미달 시 쿼리를 재작성해 재검색
- 환각(hallucination) 감지 시 재생성""")
            gr.ChatInterface(
                fn=advanced_respond,
                examples=[
                    "기가 와이파이 상품의 종류와 가격을 모두 알려줘",
                    "서비스 이용정지 사유와 해지 절차를 비교해서 설명해줘",
                    "약정 기간별로 위약금은 어떻게 달라져?",
                ],
            )

if __name__ == "__main__":
    demo.launch()

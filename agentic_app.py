import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 .env에 없습니다"

FAISS_INDEX_PATH = "faiss_index"

# ── 벡터 스토어 로드 ──────────────────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError("faiss_index가 없습니다. 먼저 embed.py를 실행하세요.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ── 에이전트가 사용할 도구 ────────────────────────────────────────
@tool
def search_doc(query: str) -> str:
    """KT 인터넷 이용약관 문서에서 관련 내용을 검색합니다.
    가격, 약관 조항, 서비스 정책 등을 찾을 때 사용하세요.
    검색어를 명확하고 구체적으로 작성할수록 좋은 결과가 나옵니다."""
    docs = retriever.invoke(query)
    if not docs:
        return "관련 문서를 찾지 못했습니다."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[문서 {i}]\n{doc.page_content}")
    return "\n\n".join(results)


# ── LLM 및 에이전트 구성 ─────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """당신은 KT 인터넷 이용약관 전문 어시스턴트입니다.
사용자의 질문에 답하기 위해 search_doc 도구를 적극 활용하세요.

행동 지침:
- 질문이 복잡하거나 여러 주제를 포함하면 검색을 여러 번 나눠서 수행하세요.
- 첫 검색 결과가 불충분하면 다른 키워드로 재검색하세요.
- 문서에서 찾은 내용만을 근거로 답변하고, 출처가 불분명한 내용은 추측임을 명시하세요.
- 가격 정보는 정확하게 인용하세요."""

memory = MemorySaver()
agent = create_react_agent(
    llm,
    tools=[search_doc],
    prompt=system_prompt,
    checkpointer=memory,
)


# ── Gradio 인터페이스 ─────────────────────────────────────────────
def respond(message: str, history: list, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    answer = result["messages"][-1].content
    return answer


with gr.Blocks(title="KT 이용약관 Agentic RAG") as demo:
    gr.Markdown("# KT 인터넷 이용약관 Agentic RAG 챗봇")
    gr.Markdown("에이전트가 스스로 검색 전략을 결정합니다. 복잡한 질문도 여러 번 검색해서 답합니다.")

    session_id = gr.State(value="default-session")

    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[session_id],
        examples=[
            "기가 와이파이 상품의 종류와 가격을 모두 알려줘",
            "서비스 이용정지 사유와 해지 절차를 비교해서 설명해줘",
            "약정 기간별로 위약금은 어떻게 달라져?",
        ],
        type="messages",
    )

if __name__ == "__main__":
    demo.launch()

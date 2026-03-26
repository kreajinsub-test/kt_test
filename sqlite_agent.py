import os
import asyncio
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 .env에 없습니다"

DB_PATH = os.path.abspath("data/w3schools/w3schools.db")

SYSTEM_PROMPT = """당신은 W3Schools 판매 데이터베이스를 분석하는 SQL 전문 어시스턴트입니다.

데이터베이스 스키마:
- Customers: 고객 정보 (CustomerID, CustomerName, ContactName, Address, City, PostalCode, Country)
- Categories: 제품 카테고리 (CategoryID, CategoryName, Description)
- Suppliers: 공급사 정보 (SupplierID, SupplierName, ContactName, Address, City, PostalCode, Country, Phone)
- Employees: 직원 정보 (EmployeeID, LastName, FirstName, BirthDate, Photo, Notes)
- Shippers: 배송사 정보 (ShipperID, ShipperName, Phone)
- Products: 제품 정보 (ProductID, ProductName, SupplierID, CategoryID, Unit, Price)
- Orders: 주문 헤더 (OrderID, CustomerID, EmployeeID, OrderDate, ShipperID)
- OrderDetails: 주문 상세 (OrderDetailID, OrderID, ProductID, Quantity)

행동 지침:
- 사용자 질문을 분석하여 적절한 SQL 쿼리를 작성하고 실행하세요.
- 복잡한 질문은 여러 쿼리로 나눠서 단계적으로 조회하세요.
- 결과를 사람이 읽기 쉬운 형태로 정리해서 답변하세요.
- 수치 데이터는 적절히 포맷팅하세요 (예: 금액은 소수점 2자리).
- 한국어로 답변하세요."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = MemorySaver()


async def create_agent_with_tools():
    """MCP 클라이언트로 SQLite 도구를 가져와 에이전트 생성"""
    client = MultiServerMCPClient(
        {
            "sqlite": {
                "command": "npx",
                "args": ["-y", "mcp-sqlite", DB_PATH],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent, client


# 전역 에이전트 (최초 1회 초기화)
_agent = None
_mcp_client = None


async def get_agent():
    global _agent, _mcp_client
    if _agent is None:
        _agent, _mcp_client = await create_agent_with_tools()
    return _agent


async def chat_async(message: str, session_id: str) -> str:
    agent = await get_agent()
    config = {"configurable": {"thread_id": session_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


def respond(message: str, history: list, session_id: str) -> str:
    return asyncio.run(chat_async(message, session_id))


# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(title="W3Schools DB SQL 에이전트") as demo:
    gr.Markdown("# W3Schools 판매 DB 질의응답 에이전트")
    gr.Markdown(
        "자연어로 질문하면 SQL을 자동 생성·실행해서 답변합니다.\n\n"
        f"**DB 경로:** `{DB_PATH}`"
    )

    session_id = gr.State(value="default-session")

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[session_id],
        examples=[
            "가장 많이 팔린 제품 상위 5개를 알려줘",
            "국가별 고객 수를 내림차순으로 보여줘",
            "카테고리별 평균 제품 가격은?",
            "주문 건수가 가장 많은 직원은 누구야?",
            "총 매출액이 가장 높은 고객 Top 3는?",
        ],
    )

if __name__ == "__main__":
    demo.launch()

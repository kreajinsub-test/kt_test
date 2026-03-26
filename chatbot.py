import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 .env에 없습니다"

FAISS_INDEX_PATH = "faiss_index"


def build_chain():
    if not os.path.exists(FAISS_INDEX_PATH):
        print("faiss_index가 없습니다. 먼저 embed.py를 실행하세요.")
        raise SystemExit(1)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 대화 기록을 고려해 검색 쿼리를 독립적으로 재작성
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

    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_with_history))
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    print("KT 인터넷 이용약관 챗봇")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("-" * 50)

    chain = build_chain()
    chat_history = []

    while True:
        try:
            user_input = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("종료합니다.")
            break

        answer = chain.invoke({"input": user_input, "chat_history": chat_history})

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        print(f"\n답변: {answer}")


if __name__ == "__main__":
    main()

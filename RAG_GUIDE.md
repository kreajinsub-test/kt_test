# KT 인터넷 이용약관 RAG 시스템 구조 문서

## 개요

이 프로젝트는 KT 인터넷 이용약관 PDF를 기반으로 3가지 방식의 RAG(Retrieval-Augmented Generation) 시스템을 구현합니다.
각 방식은 복잡도와 정확도가 다르며, 질문의 성격에 따라 적합한 방식이 달라집니다.

---

## 공통 인프라

모든 RAG 방식이 공유하는 기반 컴포넌트입니다.

```
PDF 문서
  └─ embed.py (1회 실행)
       └─ PyMuPDFLoader → 부칙 페이지 필터링 → RecursiveCharacterTextSplitter
            └─ OpenAIEmbeddings (text-embedding-3-small) → FAISS 인덱스 저장
```

```
app.py 실행 시
  └─ FAISS 인덱스 로드 → vector_store
  └─ ChatOpenAI (gpt-4o-mini) → llm
```

| 컴포넌트 | 설명 |
|---|---|
| `text-embedding-3-small` | OpenAI 임베딩 모델 (가장 저렴) |
| `gpt-4o-mini` | 답변 생성 LLM |
| `FAISS` | 로컬 벡터 DB (유사도 검색) |
| `LangSmith` | 모든 LLM 호출 추적 (환경변수로 자동 연동) |

---

## Tab 1: 일반 RAG (Basic RAG)

### 한 줄 요약
> 질문 → 1회 검색 → 답변. 가장 단순하고 빠른 방식.

### 구조 흐름

```
사용자 질문
  │
  ├─ [대화 기록 있음] → Contextualize Chain
  │     └─ LLM이 대화 맥락을 반영해 검색 쿼리 재작성
  │           └─ FAISS 검색 (k=4)
  │
  └─ [대화 기록 없음] → 원문 그대로 FAISS 검색 (k=4)
       │
       └─ 검색된 문서를 context로 QA Prompt에 주입
              └─ LLM 최종 답변 생성
```

### 핵심 코드 패턴 (LCEL 파이프라인)

```python
chain = (
    RunnablePassthrough.assign(context=RunnableLambda(retrieve_with_history))
    | qa_prompt
    | llm
    | StrOutputParser()
)
```

- `RunnablePassthrough.assign`: 입력 dict에 `context` 키를 추가하면서 나머지 키는 그대로 전달
- `RunnableLambda`: 일반 함수를 LangChain 실행 단위로 감싸는 래퍼
- `|` 연산자: LCEL(LangChain Expression Language)의 체인 연결

### 특징

| 항목 | 내용 |
|---|---|
| 검색 횟수 | 1회 고정 |
| 멀티턴 | 대화 기록 기반 쿼리 재작성으로 지원 |
| 자기 수정 | 없음 |
| 속도 | 빠름 |
| 적합한 질문 | 단순하고 명확한 1회성 질문 |

---

## Tab 2: Agentic RAG

### 한 줄 요약
> 에이전트가 스스로 검색 전략을 결정. 필요하면 여러 번 검색하고 결과를 종합.

### 구조 흐름

```
사용자 질문
  │
  └─ ReAct 에이전트 (LangGraph create_react_agent)
       │
       ├─ [Reasoning] 어떤 키워드로 검색할지 판단
       │
       ├─ [Action] search_doc(query) 도구 호출
       │     └─ FAISS 검색 (k=5)
       │
       ├─ [Observation] 검색 결과 확인
       │
       ├─ [필요시] 다른 키워드로 재검색 반복
       │
       └─ [최종 답변 생성]
```

### ReAct 패턴이란?

**Re**asoning + **Act**ing의 줄임말.
LLM이 단순히 답변을 생성하는 것이 아니라, **생각 → 행동 → 관찰** 루프를 반복하며 스스로 문제를 해결합니다.

```
Thought: 기가 와이파이 상품 가격을 찾아야 한다
Action: search_doc("기가 와이파이 가격")
Observation: [문서 1] 기가 와이파이 홈 5G... [문서 2]...
Thought: 약정 관련 내용도 필요하다
Action: search_doc("기가 와이파이 약정 요금")
Observation: ...
Final Answer: 기가 와이파이 상품은 ...
```

### 핵심 컴포넌트

```python
agent = create_react_agent(
    llm,
    tools=[search_doc],       # 사용 가능한 도구 목록
    prompt=agent_system_prompt,
    checkpointer=agent_memory, # 대화 기록 저장 (MemorySaver)
)

# 호출 시 thread_id로 대화 세션 구분
config = {"configurable": {"thread_id": session_id}}
result = agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
```

- `MemorySaver`: LangGraph의 인메모리 체크포인터. `thread_id`별로 대화 상태를 저장
- `@tool`: 함수를 LangChain 도구로 등록하는 데코레이터. docstring이 에이전트의 도구 설명이 됨

### 특징

| 항목 | 내용 |
|---|---|
| 검색 횟수 | 에이전트가 자율 결정 (1회~N회) |
| 멀티턴 | MemorySaver + thread_id로 세션별 대화 기록 유지 |
| 자기 수정 | 검색 결과 불충분 시 다른 키워드로 재검색 |
| 속도 | 느림 (LLM 호출 다수) |
| 적합한 질문 | 복잡하거나 여러 주제를 포함하는 질문 |

---

## Tab 3: Advanced RAG (CRAG - Corrective RAG)

### 한 줄 요약
> 검색된 문서의 품질과 생성된 답변의 품질을 모두 검사하고, 기준 미달이면 자동으로 수정.

### 구조 흐름 (LangGraph StateGraph)

```
START
  │
  ▼
[retrieve] ── FAISS 검색 (k=6)
  │
  ▼
[grade_documents] ── 문서별 관련성 평가 (doc_grader)
  │                   관련 없는 문서 제거
  │
  ├─ 관련 문서 있음 OR 재시도 한계 도달 ──▶ [generate]
  │
  └─ 관련 문서 없음 ──▶ [rewrite_query] ──▶ [retrieve] (루프)
                            쿼리 재작성 후 재검색

[generate] ── 문서 기반 답변 생성
  │
  ├─ 환각(Hallucination) 없음 + 답변 충분 ──▶ END
  │
  ├─ 환각 감지 ──▶ [generate] (재생성, 쿼리 변경 없음)
  │
  └─ 답변 불충분 ──▶ [rewrite_query] ──▶ [retrieve] (루프)
```

### 3개의 평가 체인 (Grader)

```python
# 1. 문서 관련성 평가: 검색된 문서가 질문과 관련 있는가?
doc_grader = prompt | llm.with_structured_output(GradeDoc)
# → GradeDoc.score: "yes" | "no"

# 2. 환각 평가: 답변이 문서에 근거하는가?
hallucination_grader = prompt | llm.with_structured_output(GradeHallucination)
# → GradeHallucination.score: "yes" | "no"

# 3. 답변 품질 평가: 답변이 질문을 충분히 해결하는가?
answer_grader = prompt | llm.with_structured_output(GradeAnswer)
# → GradeAnswer.score: "yes" | "no"
```

- `llm.with_structured_output(PydanticModel)`: LLM 출력을 Pydantic 모델로 파싱 (JSON 모드)
- `Literal["yes", "no"]`: 평가 결과를 이진값으로 강제

### 상태 관리 (TypedDict)

```python
class CRAGState(TypedDict):
    question: str          # 현재 질문 (재작성 시 업데이트됨)
    documents: List[Document]  # 검색 및 필터링된 문서
    generation: str        # 생성된 답변
    retries: int           # 재시도 횟수
    chat_history: List     # 멀티턴 대화 기록
```

모든 노드는 이 상태를 읽고 일부를 업데이트하여 반환합니다.
LangGraph가 상태를 노드 간에 자동으로 전달합니다.

### 무한 루프 방지

```python
MAX_RETRIES = 2

# 재시도 횟수가 한계에 도달하면 조건에 관계없이 종료
return "generate" if state["retries"] < MAX_RETRIES else "end"
```

### 특징

| 항목 | 내용 |
|---|---|
| 검색 횟수 | 최대 `MAX_RETRIES + 1`회 |
| 멀티턴 | chat_history를 State로 전달 |
| 자기 수정 | 문서 품질 + 환각 + 답변 품질 3단계 검사 |
| 속도 | 가장 느림 (LLM 호출 다수, 그래프 순회) |
| 적합한 질문 | 정확성이 중요하고, 잘못된 답변이 허용되지 않는 경우 |

---

## 3가지 방식 비교 요약

| | 일반 RAG | Agentic RAG | Advanced RAG (CRAG) |
|---|---|---|---|
| **검색 횟수** | 1회 | N회 (에이전트 결정) | 최대 3회 |
| **자기 수정** | 없음 | 키워드 재검색 | 문서/환각/품질 3단계 평가 |
| **멀티턴** | LCEL 수동 처리 | MemorySaver (자동) | State에 포함 |
| **핵심 기술** | LCEL 파이프라인 | ReAct 에이전트 | LangGraph StateGraph |
| **속도** | 빠름 | 느림 | 가장 느림 |
| **정확도** | 보통 | 높음 | 가장 높음 |
| **복잡도** | 낮음 | 중간 | 높음 |

---

## 핵심 기술 개념 정리

### LCEL (LangChain Expression Language)
- `|` 연산자로 컴포넌트를 체인처럼 연결
- `RunnablePassthrough`: 입력을 그대로 전달
- `RunnableLambda`: 일반 함수를 Runnable로 변환
- `.assign()`: dict에 새 키-값 추가

### LangGraph
- **StateGraph**: 노드(함수)와 엣지(연결)로 구성된 방향 그래프
- **State**: 노드 간 공유되는 데이터 구조 (TypedDict)
- **조건부 엣지**: 상태에 따라 다음 노드를 동적으로 결정
- **MemorySaver**: 그래프 실행 상태를 thread_id별로 저장

### RAG의 한계와 각 방식의 해결책

| 한계 | 일반 RAG | Agentic RAG | CRAG |
|---|---|---|---|
| 검색 쿼리가 부정확할 때 | 그대로 실패 | 다른 키워드 재시도 | 쿼리 자동 재작성 |
| 관련 없는 문서가 섞일 때 | 노이즈 포함 | 에이전트가 판단 | 문서별 관련성 평가 후 필터링 |
| LLM이 문서 없는 내용 생성 시 | 감지 불가 | 감지 불가 | 환각 감지 후 재생성 |

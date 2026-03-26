# LangChain RAG 구현 계획

## 목표
LangChain을 활용하여 로컬 문서(PDF)를 기반으로 질의응답이 가능한 멀티턴 RAG 챗봇을 구축한다.

---

## 데이터 소스
| 파일 | 형식 | 내용 |
|------|------|------|
| `data/pdf/InternetServiceToU.pdf` | PDF | KT 인터넷 이용약관 |

---

## 시스템 구조

```
[1단계: 임베딩 - 최초 1회만 실행]

PDF 로드 → 부칙 페이지 필터링 → 청크 분할
    → OpenAI text-embedding-3-small → FAISS 인덱스 저장


[2단계: 챗봇 - 매번 실행]

사용자 입력
    │
    ▼
[History-Aware Retriever]  ← chat_history
    (이전 대화 고려해 쿼리 재작성)
    │
    ▼
FAISS 유사도 검색 (Top-K)
    │
    ▼
[QA Chain] + chat_history + context
    │
    ▼
gpt-4o-mini → 답변 출력
    │
    ▼
chat_history 누적
```

---

## 파일 구조

```
kt_test/
├── embed.py        # 임베딩 및 FAISS 인덱스 생성 (최초 1회 실행)
├── chatbot.py      # 멀티턴 RAG 챗봇 (터미널 실행)
├── data/
│   └── pdf/InternetServiceToU.pdf
├── faiss_index/    # embed.py 실행 후 생성됨 (gitignore)
├── .env            # OPENAI_API_KEY
└── requirements.txt
```

### 실행 순서
```bash
# 1. 최초 1회: 임베딩 생성
python embed.py

# 2. 챗봇 실행 (faiss_index 재사용)
python chatbot.py
```

---

## 구성 요소

| 항목 | 선택 |
|------|------|
| 임베딩 | OpenAI `text-embedding-3-small` |
| 벡터 DB | FAISS (로컬 저장) |
| LLM | OpenAI `gpt-4o-mini` |
| 멀티턴 | `create_history_aware_retriever` |
| chunk_size | 800 / overlap 150 |
| Top-K | 4 |

---

## 구현 단계

### embed.py
1. `.env`에서 API 키 로드
2. `PyMuPDFLoader`로 PDF 로드
3. 부칙/시행일 위주 페이지 필터링
4. `RecursiveCharacterTextSplitter`로 청크 분할
5. `OpenAIEmbeddings(text-embedding-3-small)`로 임베딩
6. FAISS 인덱스 로컬 저장 (`faiss_index/`)

### chatbot.py
1. `.env`에서 API 키 로드
2. FAISS 인덱스 로드 (임베딩 API 호출 없음)
3. `create_history_aware_retriever` + `create_retrieval_chain` 구성
4. 터미널 루프: 입력 → chain.invoke → 출력 → chat_history 누적
5. `exit` 또는 `quit` 입력 시 종료

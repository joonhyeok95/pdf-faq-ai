본 프로젝트는 제미나이 llm연동, 로컬 gpu를 활용하여 임베딩 후 벡터디비에 저장하는 로직을 갖는다.(langchain활용)
- LLM(gemini) openapi
- VectorDB(postgreSQL)
- Embedding(huggingface/ollama) local 모델 활용

## version
- python: 3.12
- gemini model: gemini-3.1-flash-lite-preview
- huggingface embedd model: bge-m3
- ollama embedd model: all-minilm
- postgresql 16, pgvector 0.8.2


## 환경변수
프로젝트를 pull 받은 뒤 .env 파일을 생성하여 정보를 입력한다.
```
GOOGLE_API_KEY=
PGVECTOR_CONNECTION_STRING=
# PGVECTOR_CONNECTION_STRING=postgresql+psycopg2://myuser:mypassword@localhost:5432/aidb
```

## vectorDB 세팅
postgreSQL 에서 PGVector 플러그인이 존재
내장된 docker 이미지를 활용한다.(/postgresql/docker-compose.yml)

이후 아래 명령어를 수행
```
-- db 접속
docker exec -it pgvector_db psql -U myuser -d aidb
-- DB 접속 후 실행
CREATE EXTENSION vector;
-- 설치 확인 (버전이 나오면 성공!)
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## UI 툴
### streamlit 기동 방법
```
streamlit run app_streamlit.py
```
<img width="1078" height="946" alt="image" src="https://github.com/user-attachments/assets/3c96834b-558a-4a3f-8be2-092a14dc5371" />


### gradio 기동 방법
```
py app_gradio.py
```
<img width="1036" height="637" alt="image" src="https://github.com/user-attachments/assets/330d9b81-b28c-4643-a3af-1dbae41dbf14" />

## Local Embedd
ollama 명령어
```
ollama pull all-minilm
ollama list
```

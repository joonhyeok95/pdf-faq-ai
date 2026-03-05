본 프로젝트는 제미나이 llm연동, 로컬 gpu를 활용하여 임베딩 후 벡터디비에 저장하는 로직을 갖는다.
- LLM(gemini) openapi
- VectorDB(postgreSQL)
- Embedding(huggingface/ollama) local 모델 활용


streamlit 기동 방법
```
streamlit run app_streamlit.py
```


gradio 기동 방법
```
py app_gradio.py
```
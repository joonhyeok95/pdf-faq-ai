import streamlit as st
import os
import torch
import warnings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. 환경 설정 및 상수
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "False"
COLLECTION_NAME = "streamlit_pdf_rag"
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
warnings.filterwarnings("ignore")

# 2. 모델 로딩 함수 (캐싱 적용으로 매번 로드 방지)
@st.cache_resource
def init_models():
    print("🔄 모델 및 임베딩 로드 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    return embeddings, llm

def main():
    st.set_page_config(page_title="나만의 PDF AI 비서", layout="wide")
    
    # 모델 로드
    embeddings, llm = init_models()

    st.title("🚀 Local GPU RAG Assistant")
    st.markdown(f"PDF를 업로드하고 지식을 학습시키세요. (현재 장치: {'GPU' if torch.cuda.is_available() else 'CPU'})")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 사이드바: PDF 업로드 로직 ---
    with st.sidebar:
        st.header("1. 문서 업로드")
        uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")
        
        if st.button("문서 학습 시작") and uploaded_file:
            with st.spinner("GPU로 임베딩 및 DB 적재 중..."):
                # 임시 파일 저장
                with open("temp_upload.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # PDF 처리 파이프라인
                loader = PyPDFLoader("temp_upload.pdf")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                docs = loader.load_and_split(text_splitter)
                
                # pgvector 적재
                PGVector.from_documents(
                    embedding=embeddings,
                    documents=docs,
                    collection_name=COLLECTION_NAME,
                    pre_delete_collection=True,
                    use_jsonb=True
                )
                st.success(f"✅ {len(docs)}개의 지식 조각 저장 완료!")

    # --- 메인 화면: 채팅 인터페이스 ---
    st.header("2. AI와 대화하기")

    # 채팅 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("문서 내용에 대해 질문하세요!"):
        # 유저 메시지 표시 및 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("문서에서 답변을 찾는 중..."):
                # 벡터 DB 연결
                db = PGVector(
                    connection_string=PGVECTOR_CONNECTION_STRING,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings
                )
                retriever = db.as_retriever(search_kwargs={"k": 3})
                
                # RAG 체인 구성 (LCEL)
                template = """제공된 Context를 바탕으로만 답변하세요.
                Context: {context}
                Question: {question}
                Answer:"""
                prompt_template = ChatPromptTemplate.from_template(template)
                
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# 3. 진입점 설정 (윈도우 환경 필수)
if __name__ == "__main__":
    main()
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 분리된 모듈 임포트
from core.model import get_models
from core.database import get_vector_db, save_to_vector_db
from app.chain import create_rag_chain

load_dotenv()

def main():
    st.set_page_config(page_title="Local RAG Assistant", layout="wide")
    embeddings, llm = get_models()
    
    st.title("🚀 Local GPU RAG Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 사이드바: 파일 업로드 및 학습
    with st.sidebar:
        st.header("1. 문서 업로드")
        uploaded_file = st.file_uploader("PDF 선택", type="pdf")
        
        if st.button("문서 학습 시작") and uploaded_file:
            with st.spinner("학습 중..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader("temp.pdf")
                # VRAM 누락 방지를 위한 청크 사이즈 조절
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                docs = loader.load_and_split(text_splitter)
                
                save_to_vector_db(docs, embeddings)
                st.success("저장 완료!")

    # 채팅 UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("질문하세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            db = get_vector_db(embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})
            chain = create_rag_chain(retriever, llm)
            
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
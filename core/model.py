import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

# 모듈 단위에서도 환경 변수를 로드할 수 있도록 추가
load_dotenv()

@st.cache_resource
def get_models():
    # VRAM 이슈를 고려하여 장치 자동 선택
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1650 Ti 환경을 고려하여 BGE-M3 혹은 Nomic 선택 가능
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )
    
    # Gemini 모델 설정
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite-preview", temperature=0)
    
    return embeddings, llm
import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

COLLECTION_NAME = "local_tech_docs"

def main():
    # 1. 로컬 임베딩 모델 (GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

    # 2. 벡터 DB 로드
    db = PGVector(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 3. Gemini 모델 설정
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

    # 4. 프롬프트 템플릿 (RAG의 핵심)
    template = """당신은 인프라 및 개발 전문가입니다. 
    아래 제공된 문맥(Context)만을 사용하여 질문에 답변하세요. 
    답을 모르면 모른다고 말하고, 억지로 지며내지 마세요.
    답변은 친절하게 한국어로 작성하세요.

    #문맥:
    {context}

    #질문:
    {question}

    #답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. LangChain Expression Language (LCEL) 체인 생성
    # 
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. 실행
    print("\n🤖 RAG 시스템 준비 완료!")
    query = "pgvector가 도커 환경에서 어떤 역할을 하나요?"
    
    print(f"\n❓ 질문: {query}")
    print("⏳ 답변 생성 중...")
    
    response = rag_chain.invoke(query)
    
    print("\n✨ Gemini의 답변:")
    print(response)

if __name__ == "__main__":
    main()
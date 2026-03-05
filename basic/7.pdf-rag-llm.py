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

# 보안 및 환경 설정
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "False"
warnings.filterwarnings("ignore")

COLLECTION_NAME = "pdf_knowledge_base"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 로컬 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

    # 2. PDF 로드 및 텍스트 분할 (Ingestion)
    # 'data/manual.pdf' 파일이 있다고 가정합니다.
    pdf_path = "C:\\dev\\AI\\python\\필립스_커피머신_제품사용가이드.pdf" 
    if not os.path.exists(pdf_path):
        print(f"❌ '{pdf_path}' 파일을 찾을 수 없습니다. 폴더와 파일을 확인해주세요.")
        return

    print(f"📄 PDF 읽는 중: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 문서를 의미 있는 단위로 자르기 (Chunking)
    # BGE-M3의 최대 토큰 길이를 고려하여 500~1000자 정도로 자르는 것이 좋습니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    print(f"✂️ 문서를 {len(docs)}개의 조각으로 나누었습니다.")

    # 3. pgvector에 적재
    print("📥 벡터 DB에 저장 중 (GPU 가속)...")
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True # 기존 데이터 초기화 (필요시 False)
    )

    # 4. RAG 체인 구성
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 업로드된 문서의 내용을 바탕으로 답변하는 전문가입니다.
    제공된 문맥(Context)을 바탕으로 질문에 정확하게 답변하세요.
    
    #문맥:
    {context}
    
    #질문:
    {question}
    
    #답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 실행
    print("\n✅ PDF 학습 완료! 질문을 입력하세요.")
    user_query = "이 문서의 핵심 내용을 요약해줘" # PDF 내용에 맞는 질문을 해보세요
    
    print(f"\n❓ 질문: {user_query}")
    response = rag_chain.invoke(user_query)
    
    print("\n✨ Gemini의 답변:")
    print(response)

if __name__ == "__main__":
    main()
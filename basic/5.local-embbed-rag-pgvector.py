import os
from dotenv import load_dotenv
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

# .env 파일로부터 환경변수 로드
load_dotenv()
# 1. DB 연결 설정 (Docker Compose 설정에 맞게 수정)
# CONNECTION_STRING = os.getenv("CONNECTION_STRING")  # env 설정해서 생략가능
COLLECTION_NAME = "local_tech_docs"

def main():
    # 2. 로컬 임베딩 모델 로드 (GPU 활용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

    # 3. 테스트용 데이터 준비 (문서 객체 생성)
    data = [
        "Docker Compose를 이용하면 여러 컨테이너를 한 번에 관리할 수 있습니다.",
        "pgvector는 PostgreSQL에서 벡터 유사도 검색을 가능하게 해주는 확장 기능입니다.",
        "BGE-M3는 다국어 지원이 뛰어난 고성능 오픈소스 임베딩 모델입니다.",
        "Full-stack 개발은 프론트엔드와 백엔드를 모두 다루는 것을 의미합니다."
    ]
    docs = [Document(page_content=text, metadata={"source": "manual"}) for text in data]

    # 4. pgvector에 데이터 적재
    # 최초 실행 시 테이블을 생성하고 데이터를 벡터화하여 저장합니다.
    print(f"--- pgvector 적재 시작 ({len(docs)}개 문서) ---")
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        # connection_string=CONNECTION_STRING, # env 설정해서 생략가능
        pre_delete_collection=True # 테스트를 위해 실행 시마다 초기화 (필요시 False)
    )
    print("--- 적재 완료 ---")

    # 5. 유사도 검색 테스트
    query = "컨테이너 오케스트레이션과 도커에 대해 알려줘"
    print(f"\n🔍 검색 질문: {query}")
    
    # 상위 2개의 가장 유사한 문서 검색
    results = db.similarity_search(query, k=2)

    print("\n--- 검색 결과 ---")
    for i, res in enumerate(results):
        print(f"[{i+1}] {res.page_content}")

if __name__ == "__main__":
    # 윈도우 멀티프로세싱 에러 방지
    main()
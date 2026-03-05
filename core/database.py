import os
# from langchain_community.vectorstores import PGVector # deprecated..
from langchain_postgres.vectorstores import PGVector as PGVectorStore # 별칭 사용 가능
from dotenv import load_dotenv

# 모듈 단위에서도 환경 변수를 로드할 수 있도록 추가
load_dotenv()

COLLECTION_NAME = "streamlit_pdf_rag"
CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

def get_vector_db(embeddings):
    return PGVectorStore(
        connection=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        use_jsonb=True
    )

def save_to_vector_db(docs, embeddings):
    PGVectorStore.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        use_jsonb=True
    )
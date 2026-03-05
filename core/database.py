import os
from langchain_community.vectorstores import PGVector
from dotenv import load_dotenv

# 모듈 단위에서도 환경 변수를 로드할 수 있도록 추가
load_dotenv()

COLLECTION_NAME = "streamlit_pdf_rag"
CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

def get_vector_db(embeddings):
    return PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        use_jsonb=True
    )

def save_to_vector_db(docs, embeddings):
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        use_jsonb=True
    )
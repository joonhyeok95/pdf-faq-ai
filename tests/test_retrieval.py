import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트 경로를 path에 추가하여 app, core 모듈을 불러올 수 있게 함
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from core.model import get_models
from core.database import get_vector_db

def test_search_samsung_experience():
    """삼성전자 관련 경력이 정상적으로 검색되는지 테스트"""
    load_dotenv()
    
    print("\n🔍 테스트 시작: 모델 및 DB 연결 중...")
    embeddings, _ = get_models()
    vector_db = get_vector_db(embeddings)
    
    # 1. 검색 테스트 (가장 중요한 삼성전자 키워드)
    query = "삼성전자 DS 부문 프로젝트 경험에 대해 알려줘"
    print(f"❓ 질문: {query}")
    
    # 상위 3개의 관련 문서 조각을 가져옴
    docs = vector_db.similarity_search(query, k=3)
    
    print(f"📚 검색된 문서 수: {len(docs)}개")
    
    # 2. 검증 로직
    if len(docs) > 0:
        print("✅ 검색 결과 성공!")
        for i, doc in enumerate(docs):
            print(f"--- [결과 {i+1}] ---")
            # 내용 중 일부 출력 (삼성전자가 포함되어 있는지 확인)
            print(doc.page_content[:200].replace('\n', ' ')) 
            print(f"메타데이터: {doc.metadata}")
    else:
        print("❌ 검색 결과가 없습니다. DB 적재 상태를 확인하세요.")
        assert False, "검색 결과가 0건입니다."

    # 3. 특정 키워드 포함 여부 단언(Assert)
    # 실제 문서에 포함된 핵심 단어를 사용하여 테스트 성공 여부 판단
    found = any("삼성" in doc.page_content for doc in docs)
    assert found, "검색된 문서에 '삼성' 키워드가 포함되어 있지 않습니다."
    print("✨ 테스트 통과: 데이터가 정상적으로 검색됩니다.")

if __name__ == "__main__":
    test_search_samsung_experience()
import os
import warnings
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    # 1. 경고 무시 (선택 사항)
    warnings.filterwarnings("ignore")

    # 2. 로컬 임베딩 모델 설정
    # 모델명: BAAI/bge-m3 (다국어 지원 및 한국어 성능 우수)
    model_name = "BAAI/bge-m3"
    model_kwargs = {
            'device': 'cuda',  # GPU가 있다면 'cuda'로 변경 
        }
    encode_kwargs = {'normalize_embeddings': True}

    print(f"--- 모델 로딩 시작: {model_name} ---")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("--- 모델 로딩 완료 ---")

    # 3. 임베딩 테스트
    text = "풀스택 개발자를 위한 로컬 임베딩 테스트입니다."
    query_result = hf_embeddings.embed_query(text)

    # 4. 결과 확인
    print(f"\n텍스트: {text}")
    print(f"임베딩 차원: {len(query_result)}") # BGE-M3는 1024차원입니다.
    print(f"샘플 벡터(앞부분 5개): {query_result[:5]}")

    # 5. 문장 간 유사도 비교 테스트
    sentence1 = "나는 파이썬을 공부하고 있어요."
    sentence2 = "파이썬 학습 중입니다."
    sentence3 = "오늘 점심 메뉴는 돈까스입니다."

    vec1 = hf_embeddings.embed_query(sentence1)
    vec2 = hf_embeddings.embed_query(sentence2)
    vec3 = hf_embeddings.embed_query(sentence3)

    # 단순 코사인 유사도 계산 (벡터 라이브러리 없이 수동 확인용)
    import numpy as np
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print(f"\n'문장1' vs '문장2' 유사도: {cosine_similarity(vec1, vec2):.4f}")
    print(f"'문장1' vs '문장3' 유사도: {cosine_similarity(vec1, vec3):.4f}")


# 5. 윈도우 필수 진입점 설정 (가장 중요!)
if __name__ == '__main__':
    main()
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. API 키 설정 (환경 변수로 관리하는 것을 추천합니다)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAEMUqW5Kva9GN9EcmqFIZ92PYtWSRIZRQ"

# 2. 모델 초기화 (무료 티어 모델: gemini-1.5-flash 권장)
# flash 모델이 속도가 빠르고 무료 토큰 한도가 넉넉합니다.
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7,  # 창의성 조절 (0~1)
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 3. 간단한 호출 테스트
response = llm.invoke("풀스택 개발자가 AI를 공부할 때 가장 중요한 것 3가지만 알려줘.")

print("--- Gemini의 답변 ---")
print(response.content)
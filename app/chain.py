import os
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from app.prompt import get_rag_prompt
from typing import List, Annotated, TypedDict
import operator

# 1. 상태 정의 (State)
class GraphState(TypedDict):
    question: str
    context: List
    answer: str
    relevance: str  # 'yes' or 'no'
    loop_count: Annotated[int, operator.add] # 루프 횟수 누적 추적

# 2. 노드 함수 정의
def retrieve_node(state, retriever):
    print("--- [Node] Retrieve ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"context": documents, "question": question}

def evaluate_node(state, llm):
    print(f"--- [Node] Evaluate (Attempt {state.get('loop_count', 0) + 1}) ---")
    question = state["question"]
    context = "\n".join([doc.page_content for doc in state["context"]])
    
    if not context.strip(): # 검색 결과가 아예 없는 경우
        return {"relevance": "no", "loop_count": 1}

    eval_prompt = ChatPromptTemplate.from_template(
        "너는 문서의 유용성을 판별하는 전문가야. 질문: {question}\n문서: {context}\n"
        "이 문서가 질문에 답하기에 충분한 정보를 포함하고 있으면 'yes', 아니면 'no'라고 답해."
    )
    eval_chain = eval_prompt | llm | StrOutputParser()
    relevance = eval_chain.invoke({"context": context, "question": question}).strip().lower()
    
    return {"relevance": "yes" if "yes" in relevance else "no", "loop_count": 1}

def rewrite_node(state, llm):
    print("--- [Node] Rewrite Query ---")
    question = state["question"]
    
    rewrite_prompt = ChatPromptTemplate.from_template(
        "사용자의 질문을 벡터 DB 검색에 최적화되도록 더 구체적으로 재작성해줘.\n"
        "원래 질문: {question}\n"
        "재작성된 질문:"
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    new_question = rewrite_chain.invoke({"question": question})
    return {"question": new_question}

def generate_node(state, llm):
    print("--- [Node] Generate Answer ---")
    prompt_template = get_rag_prompt()
    chain = prompt_template | llm | StrOutputParser()
    
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": response}

def decide_to_generate(state):
    """다음 노드를 결정하는 조건부 에지 함수"""
    print(f"--- [Decision] Checking Relevance & Loop Count ({state['loop_count']}) ---")
    
    if state["relevance"] == "yes":
        return "generate"
    
    # 재시도 횟수 제한 (예: 최대 2번만 Rewrite 허용)
    if state["loop_count"] >= 2:
        print("--- [Decision] Max loops reached. Forcing generation or fallback ---")
        return "generate" # 혹은 "web_search" 노드로 연결 가능
        
    return "rewrite"

# 3. 그래프 구성 함수
def create_langgraph_workflow(retriever, llm):
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    workflow.add_node("evaluate", lambda state: evaluate_node(state, llm))
    workflow.add_node("rewrite", lambda state: rewrite_node(state, llm))
    workflow.add_node("generate", lambda state: generate_node(state, llm))

    # 에지 연결 (흐름 정의)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")
    
    # 조건부 에지 (Evaluator 로직)
    workflow.add_conditional_edges(
        "evaluate",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    workflow.add_edge("rewrite", "retrieve") # 다시 검색하러 이동
    workflow.add_edge("generate", END)

    return workflow.compile()
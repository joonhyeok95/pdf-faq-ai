import os
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from app.prompt import get_rag_prompt

# 1. 상태 정의 (State)
class GraphState(TypedDict):
    question: str
    context: List
    answer: str
    relevance: str  # 'yes' or 'no'

# 2. 노드 함수 정의
def retrieve_node(state, retriever):
    print("--- [Node] Retrieve ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"context": documents, "question": question}

def evaluate_node(state, llm):
    print("--- [Node] Evaluate (Check Relevance) ---")
    question = state["question"]
    context = "\n".join([doc.page_content for doc in state["context"]])
    
    # 이진 분류를 위한 프롬프트
    eval_prompt = ChatPromptTemplate.from_template(
        "너는 검색된 문서가 사용자의 질문과 관련이 있는지 평가하는 평가관이야.\n"
        "문서 내용: {context}\n"
        "사용자 질문: {question}\n"
        "관련이 있다면 'yes', 없다면 'no'라고만 답해."
    )
    eval_chain = eval_prompt | llm | StrOutputParser()
    relevance = eval_chain.invoke({"context": context, "question": question}).strip().lower()
    
    return {"relevance": "yes" if "yes" in relevance else "no"}

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
        lambda state: state["relevance"],
        {
            "yes": "generate",  # 관련 있으면 답변 생성
            "no": "rewrite"     # 관련 없으면 재작성
        }
    )
    workflow.add_edge("rewrite", "retrieve") # 다시 검색하러 이동
    workflow.add_edge("generate", END)

    return workflow.compile()
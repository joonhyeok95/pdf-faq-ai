from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt():
    template = """제공된 Context를 바탕으로만 답변하세요. 
    Context 내에 답변이 없다면 "문서에 관련 내용이 없습니다"라고 답하세요.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    return ChatPromptTemplate.from_template(template)
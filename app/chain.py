from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.prompt import get_rag_prompt

def create_rag_chain(retriever, llm):
    prompt_template = get_rag_prompt()
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain
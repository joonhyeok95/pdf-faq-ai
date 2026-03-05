import os
import torch
import gradio as gr
import warnings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. 환경 설정 및 보안 패치
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_REMOVAL"] = "False"
COLLECTION_NAME = "gradio_pdf_rag"
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
warnings.filterwarnings("ignore")

# 글로벌 변수 선언 (나중에 main에서 할당)
embeddings = None
llm = None

def process_pdf(file):
    if file is None: return "파일이 없습니다."
    loader = PyPDFLoader(file.name)
    docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100))
    PGVector.from_documents(
        connection_string=PGVECTOR_CONNECTION_STRING,
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True,
        use_jsonb=True
    )
    return f"✅ {len(docs)}개 조각 학습 완료!"

def predict(message, history):
    db = PGVector(collection_name=COLLECTION_NAME, 
                  embedding_function=embeddings, 
                  connection_string=PGVECTOR_CONNECTION_STRING)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:")
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(message)

# 2. 모든 실행 로직을 이 블록 안으로 격리!
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 로딩을 여기 안에서 해야 윈도우 멀티프로세싱 에러가 안 납니다.
    print(f"🚀 모델 로딩 중... ({device})")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

    # UI 구성
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 BGE-M3 + pgvector 로컬 RAG")
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="PDF 업로드")
                upload_btn = gr.Button("학습 시작", variant="primary")
                status = gr.Textbox(label="상태", interactive=False)
            with gr.Column(scale=3):
                gr.ChatInterface(fn=predict, title="AI 문서 비서")
        
        upload_btn.click(process_pdf, inputs=[file_input], outputs=[status])

    print("✨ Gradio 서버 가동!")
    demo.launch()
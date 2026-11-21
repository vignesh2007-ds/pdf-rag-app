import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

client = genai.Client(api_key=GEMINI_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_API_KEY,
    model=EMBEDDING_MODEL
)

vector_store = None
pdf_name = None


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def create_vector_store(pdf_path: str, original_name: str):
    global vector_store, pdf_name

    text = extract_pdf_text(pdf_path)

    if not text.strip():
        raise ValueError("No readable text found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )

    chunks = splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    pdf_name = original_name

    return {
        "pdf_name": pdf_name,
        "pages": len(reader_pages(pdf_path)),
        "chunks": len(chunks)
    }


def reader_pages(pdf_path: str):
    return PdfReader(pdf_path).pages


def answer_question(question: str) -> dict:
    global vector_store, pdf_name

    if vector_store is None:
        raise ValueError("Upload a PDF first.")

    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[Source {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    prompt = f"""
You are a helpful assistant answering questions strictly from the PDF context.

Rules:
1. Use only the provided context.
2. If the answer is not present, say: "The answer is not available in the uploaded PDF."
3. Be clear and concise.
4. Mention source numbers when useful.

PDF name: {pdf_name}

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": [doc.page_content[:350] for doc in docs]
    }

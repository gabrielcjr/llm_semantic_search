import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

for k in (
    "PDF_PATH",
    "PGVECTOR_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "GOOGLE_API_KEY",
):
    if not os.getenv(k):
        raise RuntimeError(f"Missing environment variable: {k}")

current_dir = Path(__file__).parent.parent
pdf_path = current_dir / os.getenv("PDF_PATH")

docs = PyPDFLoader(str(pdf_path)).load()

splits = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, add_start_index=False
).split_documents(docs)

if not splits:
    raise SystemExit(0)

enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
    )
    for d in splits
]

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    api_key=os.getenv("GOOGLE_API_KEY"),
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

def ingest_pdf():
    store.add_documents(enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()

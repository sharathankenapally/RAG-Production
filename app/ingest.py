import os
import re
import uuid
import sys
import ollama
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, SparseVector
from fastembed import SparseTextEmbedding

# Ensure current directory is in PYTHONPATH to resolve app imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM
except ModuleNotFoundError:
    from config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM

qdrant = QdrantClient(url=QDRANT_URL)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def extract_pdf_text(pdf_path):
    text_parts = []
    print(f"Opening PDF: {os.path.basename(pdf_path)}...", flush=True)
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages to extract: {total_pages}", flush=True)
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            if (i + 1) % 50 == 0 or i + 1 == total_pages:
                print(f"Extracted text from {i + 1}/{total_pages} pages...", flush=True)
    return "\n".join(text_parts)

def chunk_text(text, max_words=100, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], []
    word_count = 0
    for s in sentences:
        current.append(s)
        word_count += len(s.split())
        if word_count >= max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            word_count = sum(len(c.split()) for c in current)
    if current:
        chunks.append(" ".join(current))
    return chunks

def embed_text(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def setup_collection():
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        sparse_vectors_config={
            "sparse-text": SparseVectorParams()
        }
    )

def ingest_folder(folder="docs", raw_folder="raw"):
    setup_collection()
    points = []
    batch_size = 50
    total_ingested = 0

    # Ensure docs directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Handle plain .txt files
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue
        print(f"Processing text file: {filename}...", flush=True)
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        print(f"Total chunks generated: {len(chunks)}", flush=True)
        for idx, chunk in enumerate(chunks):
            dense_vector = embed_text(chunk)
            sparse_emb = list(sparse_model.embed([chunk]))[0]
            vector_dict = {
                "": dense_vector,
                "sparse-text": SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                )
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector_dict,
                                       payload={"text": chunk, "source": filename}))
            
            if len(points) >= batch_size:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                total_ingested += len(points)
                print(f"Upserted batch of {len(points)} chunks (Total: {total_ingested})...", flush=True)
                points = []

    # Handle PDFs directly
    if os.path.exists(raw_folder):
        for filename in os.listdir(raw_folder):
            if not filename.endswith(".pdf"):
                continue
            print(f"Processing PDF file: {filename}...", flush=True)
            text = extract_pdf_text(os.path.join(raw_folder, filename))
            chunks = chunk_text(text)
            print(f"Total chunks generated: {len(chunks)}", flush=True)
            for idx, chunk in enumerate(chunks):
                dense_vector = embed_text(chunk)
                sparse_emb = list(sparse_model.embed([chunk]))[0]
                vector_dict = {
                    "": dense_vector,
                    "sparse-text": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    )
                }
                points.append(PointStruct(id=str(uuid.uuid4()), vector=vector_dict,
                                           payload={"text": chunk, "source": filename}))
                
                if len(points) >= batch_size:
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    total_ingested += len(points)
                    print(f"Upserted batch of {len(points)} chunks (Total: {total_ingested})...", flush=True)
                    points = []

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total_ingested += len(points)
        print(f"Upserted final batch of {len(points)} chunks.", flush=True)

    print(f"Ingestion complete. Ingested {total_ingested} chunks total.", flush=True)

def ingest_pdf_bytes(file_bytes: bytes, filename: str):
    import io
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    text = "\n".join(text_parts)
    
    chunks = chunk_text(text)
    points = []
    
    # Ensure collection exists (do not delete old collections, we append!)
    if not qdrant.collection_exists(COLLECTION_NAME):
        setup_collection()
        
    for chunk in chunks:
        dense_vector = embed_text(chunk)
        sparse_emb = list(sparse_model.embed([chunk]))[0]
        vector_dict = {
            "": dense_vector,
            "sparse-text": SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist()
            )
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector_dict,
            payload={"text": chunk, "source": filename}
        ))
        
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} chunks from uploaded file: {filename}", flush=True)

if __name__ == "__main__":
    # If run directly, ingest from the relative docs directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "docs")
    raw_dir = os.path.join(base_dir, "raw")
    ingest_folder(folder=docs_dir, raw_folder=raw_dir)

import os
import sys
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchAny
# pyrefly: ignore [missing-import]
from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Ensure current directory is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL
except ModuleNotFoundError:
    from config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL

qdrant = QdrantClient(url=QDRANT_URL)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

def retrieve_contexts(query: str, top_k: int = 3, filters: list = None) -> list:
    """
    Retrieves candidate context passages using a hybrid (Dense + BM25) Qdrant query,
    reranks them using a local Cross-Encoder (MiniLM), and returns the top_k contexts.
    """
    try:
        # Check if collection exists
        collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            print(f"Collection '{COLLECTION_NAME}' does not exist yet.")
            return []

        # 1. Generate query embeddings
        # A. Dense embedding
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_vector = response["embedding"]
        
        # B. Sparse (BM25) embedding
        query_sparse = list(sparse_model.embed([query]))[0]
        
        # Build filter condition if document filters are supplied
        filter_condition = None
        if filters:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=filters)
                    )
                ]
            )
            
        # 1.5 Check similarity threshold of top dense match to filter off-topic queries
        dense_check = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=filter_condition,
            limit=1
        ).points
        
        DENSE_THRESHOLD = 0.50
        if not dense_check or dense_check[0].score < DENSE_THRESHOLD:
            score_val = dense_check[0].score if dense_check else 0.0
            print(f"Top similarity score ({score_val:.4f}) is below threshold ({DENSE_THRESHOLD}). Blocking as off-topic.", flush=True)
            return []
        
        # 2. Retrieve top 20 candidate points from Qdrant using RRF Hybrid Search
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="",
                    limit=20,
                    filter=filter_condition
                ),
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist()
                    ),
                    using="sparse-text",
                    limit=20,
                    filter=filter_condition
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=20
        ).points
        
        if not search_result:
            return []

        # 3. Rerank candidates using the local Cross-Encoder model
        candidate_texts = [
            p.payload["text"] 
            for p in search_result 
            if p.payload and "text" in p.payload
        ]
        
        if candidate_texts:
            # Score each candidate chunk
            scores = list(reranker.rerank(query, candidate_texts))
            # Pair search results with their cross-encoder scores
            paired = list(zip(search_result, scores))
            # Sort descending by score
            paired.sort(key=lambda x: x[1], reverse=True)
            # Slice the top_k results
            search_result = [item[0] for item in paired[:top_k]]
        
        # 4. Extract text from payloads
        contexts = [
            hit.payload["text"] 
            for hit in search_result 
            if hit.payload and "text" in hit.payload
        ]
        return contexts
    except Exception as e:
        print(f"Error retrieving contexts: {e}")
        return []

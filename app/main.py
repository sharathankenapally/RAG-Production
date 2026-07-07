import os
import sys
import json
import redis
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile

# Ensure current directory is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.config import REDIS_HOST, REDIS_PORT, COLLECTION_NAME
    from app.models import QueryRequest, QueryResponse
    from app.ingest import ingest_folder, ingest_pdf_bytes
    from app.retrieval import retrieve_contexts
    from app.generate import generate_answer, rephrase_query, is_query_safe, evaluate_response
except ModuleNotFoundError:
    from config import REDIS_HOST, REDIS_PORT, COLLECTION_NAME
    from models import QueryRequest, QueryResponse
    from ingest import ingest_folder, ingest_pdf_bytes
    from retrieval import retrieve_contexts
    from generate import generate_answer, rephrase_query, is_query_safe, evaluate_response

app = FastAPI(
    title="Production RAG API",
    description="FastAPI RAG application with Qdrant vector database, Ollama LLM, and Redis caching."
)

# Initialize Redis client safely
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    redis_enabled = True
    print("Redis connected successfully.")
except Exception as e:
    redis_client = None
    redis_enabled = False
    print(f"Redis connection failed: {e}. Caching is disabled.")

@app.get("/health")
def health_check():
    """
    Returns health status of the application and its storage connections.
    """
    health = {
        "status": "healthy",
        "redis_connected": redis_enabled,
        "qdrant_status": "unknown"
    }
    
    # Verify Qdrant connection
    try:
        from qdrant_client import QdrantClient
        from app.config import QDRANT_URL
        qdrant = QdrantClient(url=QDRANT_URL)
        qdrant.get_collections()
        health["qdrant_status"] = "connected"
    except Exception as e:
        health["qdrant_status"] = f"error: {e}"
        health["status"] = "unhealthy"
        
    return health

@app.post("/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Asynchronously parses, chunks, embeds, and uploads documents to Qdrant.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "docs")
    raw_dir = os.path.join(base_dir, "raw")
    
    # Ensure folders exist
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    background_tasks.add_task(ingest_folder, folder=docs_dir, raw_folder=raw_dir)
    return {"message": "Document ingestion started in the background."}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Queries document store to retrieve facts and generate a cached response.
    """
    query_str = request.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_key = f"rag:session:{request.session_id}" if request.session_id else None
    history = []

    # 1. Fetch conversation history from Redis
    if session_key and redis_enabled and redis_client:
        try:
            hist_val = redis_client.get(session_key)
            if hist_val:
                history = json.loads(hist_val)
        except Exception as e:
            print(f"Error reading session history from Redis: {e}")

    # 2. Look up query response in cache (only if NOT in a session to avoid collision)
    if not request.session_id and redis_enabled and redis_client:
        cache_key = f"rag:cache:{query_str.lower()}"
        try:
            cached_val = redis_client.get(cache_key)
            if cached_val:
                cached_data = json.loads(cached_val)
                return QueryResponse(
                    answer=cached_data["answer"],
                    contexts=cached_data["contexts"],
                    cached=True,
                    evaluation=cached_data.get("evaluation")
                )
        except Exception as e:
            print(f"Error reading from Redis cache: {e}")

    # 3. Rephrase query using LLM if history exists
    search_query = query_str
    if history:
        search_query = rephrase_query(history, query_str)

    # 4. Retrieve matching document chunks
    contexts = retrieve_contexts(search_query, top_k=request.top_k, filters=request.filters)
    
    # 5. Generate answer using Ollama LLM
    if not contexts:
        answer = "I cannot find the answer in the provided documents."
        evaluation = {
            "faithfulness": 0,
            "relevance": 0,
            "reason": "Query blocked by similarity threshold (no matching documents found)."
        }
    else:
        answer = generate_answer(query_str, contexts, chat_history=history)
        if answer.startswith("Error generating answer"):
            raise HTTPException(status_code=500, detail=answer)
        
        # 5.5 Run local RAG evaluation
        evaluation = evaluate_response(query_str, contexts, answer)
    
    # 6. Save results to Redis session history or standard cache
    if request.session_id and redis_enabled and redis_client:
        try:
            history.append({"role": "user", "content": query_str})
            history.append({"role": "assistant", "content": answer})
            # Keep history capped to last 10 messages (5 turns)
            redis_client.setex(session_key, 3600, json.dumps(history[-10:]))
        except Exception as e:
            print(f"Error writing session history to Redis: {e}")
    elif redis_enabled and redis_client:
        # Standard query cache (1 hour TTL)
        try:
            cache_key = f"rag:cache:{query_str.lower()}"
            cache_data = {
                "answer": answer,
                "contexts": contexts,
                "evaluation": evaluation
            }
            redis_client.setex(cache_key, 3600, json.dumps(cache_data))
        except Exception as e:
            print(f"Error writing to Redis cache: {e}")

    return QueryResponse(
        answer=answer,
        contexts=contexts,
        cached=False,
        evaluation=evaluation
    )

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, extracts its contents, chunks, encodes, 
    and inserts the documents into Qdrant.
    """
    try:
        file_content = await file.read()
        
        # Save file to raw/ folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_dir = os.path.join(base_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_path = os.path.join(raw_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
            
        ingest_pdf_bytes(file_content, file.filename)
        # Flush Redis cache on new uploads to ensure fresh context
        if redis_enabled and redis_client:
            redis_client.flushall()
            print("Flushed Redis cache on new document upload.")
        return {"message": f"Successfully ingested {file.filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os
import uvicorn
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
PORT = int(os.getenv("PORT", 8000))

if not DB_URL:
    raise RuntimeError("DATABASE_URL environment variable is missing.")

# --- DATABASE SETUP ---
def get_db_connection():
    """Establishes a connection to the remote Neon PostgreSQL database."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn

def initialize_database():
    """Injects the vector extension and creates the RAG table on cloud boot."""
    print("[SYSTEM] Waking up Neon database...")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cloud_documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(3) NOT NULL
                );
            """)
            print("[SYSTEM] Cloud pgvector database initialized successfully.")

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs immediately AFTER Uvicorn binds the port, preventing Render timeouts.
    try:
        initialize_database()
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
    yield
    # Anything here runs on server shutdown

app = FastAPI(title="Cloud pgvector RAG API", lifespan=lifespan)

# --- API MODELS ---
class DocumentPayload(BaseModel):
    content: str
    embedding: list[float]  

class SearchQuery(BaseModel):
    embedding: list[float]  
    top_k: int = 2

# --- API ROUTES ---
@app.post("/api/v1/ingest")
def ingest_document(payload: DocumentPayload):
    if len(payload.embedding) != 3:
        raise HTTPException(status_code=400, detail="Embedding must be 3 dimensions.")
    
    with get_db_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cloud_documents (content, embedding) VALUES (%s, %s) RETURNING id;",
                (payload.content, payload.embedding)
            )
            doc_id = cur.fetchone()[0]
            
    return {"status": "success", "inserted_id": doc_id, "content": payload.content}

@app.post("/api/v1/search")
def vector_search(query: SearchQuery):
    if len(query.embedding) != 3:
        raise HTTPException(status_code=400, detail="Query embedding must be 3 dimensions.")
    
    with get_db_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, embedding <-> %s AS distance
                FROM cloud_documents
                ORDER BY distance ASC
                LIMIT %s;
            """, (query.embedding, query.top_k))
            
            results = cur.fetchall()
            
    return {
        "status": "success",
        "matches": [
            {"id": row[0], "content": row[1], "distance": round(row[2], 4)}
            for row in results
        ]
    }

if __name__ == "__main__":
    uvicorn.run("project_100code1:app", host="0.0.0.0", port=PORT, reload=True)

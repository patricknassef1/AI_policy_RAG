"""
app.py - FastAPI RAG API
"""

import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv


load_dotenv()


# ==============================
# CONFIG
# ==============================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


MODEL_ID = os.getenv(
    "GROQ_MODEL_ID",
    "llama-3.3-70b-versatile"
)


TOP_K = 4

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")

if not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")



# ==============================
# CLIENTS
# ==============================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


groq_client = Groq(
    api_key=GROQ_API_KEY
)


# Lazy model loading
embedder = None


def get_embedder():

    global embedder

    if embedder is None:

        print("Loading embedding model...")

        embedder = SentenceTransformer(
            EMBED_MODEL,
            device="cpu"
        )

        print("Embedding model loaded")

    return embedder



# ==============================
# PROMPT
# ==============================

SYSTEM_PROMPT = """
You are a helpful assistant answering questions using only the provided documents.

Rules:

1. Use only the context provided.
2. The user may use different wording than the document.
3. Match meaning, not exact keywords.
4. If the answer exists in the context, explain it clearly.
5. Only say "I couldn't find this information in the documents" if the context does not contain the answer.
6. Never invent information.
"""



# ==============================
# APP
# ==============================

app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ==============================
# SCHEMAS
# ==============================

class AskRequest(BaseModel):

    question: str

    history: Optional[List[dict]] = None



class AskResponse(BaseModel):

    answer: str

    sources: List[str]



# ==============================
# EMBEDDING
# ==============================

def embed_query(question: str):

    model = get_embedder()

    embedding = model.encode(
        "query: " + question,
        normalize_embeddings=True
    )


    return embedding.tolist()



# ==============================
# RETRIEVAL
# ==============================

def retrieve_context(
    query: str,
    top_k=TOP_K
):

    query_embedding = embed_query(query)


    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k
        }
    ).execute()


    chunks = result.data or []


    print(
        f"Retrieved chunks: {len(chunks)}"
    )


    return chunks



# ==============================
# BUILD PROMPT
# ==============================

def build_messages(
    question,
    chunks,
    history=None
):

    context = "\n\n".join(
        [
            f"""
Source:
{chunk["metadata"].get("source")}

Content:
{chunk["content"]}
"""
            for chunk in chunks
        ]
    )


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


    if history:

        messages.extend(
            [
                msg
                for msg in history
                if msg.get("role")
                and msg.get("content")
            ]
        )


    messages.append(
        {
            "role": "user",
            "content": f"""
Context:

{context}


Question:

{question}


Answer:
"""
        }
    )


    return messages



# ==============================
# ROUTES
# ==============================


@app.get("/")
def health():

    return {
        "status": "ok",
        "message": "RAG API running"
    }



@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    try:

        chunks = retrieve_context(
            request.question
        )


        if not chunks:

            return AskResponse(
                answer="I couldn't find this information in the documents.",
                sources=[]
            )



        messages = build_messages(
            request.question,
            chunks,
            request.history
        )



        sources = list(
            {
                c["metadata"].get(
                    "source",
                    "unknown"
                )
                for c in chunks
            }
        )



        response = groq_client.chat.completions.create(

            model=MODEL_ID,

            messages=messages,

            max_tokens=800,

            temperature=0.3
        )



        answer = (
            response
            .choices[0]
            .message
            .content
        )


        return AskResponse(
            answer=answer,
            sources=sources
        )



    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )


        return AskResponse(

            answer=f"Error: {str(e)}",

            sources=[]

        )



# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                8000
            )
        )

    )

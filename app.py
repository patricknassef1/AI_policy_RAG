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


# ---------------- CONFIG ----------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_ID = os.getenv(
    "GROQ_MODEL_ID",
    "llama-3.3-70b-versatile"
)


if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_KEY environment variables"
    )


if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY environment variable"
    )


TOP_K = 4
EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------- CLIENTS ----------------

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


embedder = SentenceTransformer(
    EMBED_MODEL
)


client = Groq(
    api_key=GROQ_API_KEY
)


# ---------------- PROMPT ----------------

SYSTEM_PROMPT = """
You are a helpful assistant answering questions based only on the provided documents.

Instructions:
- Use the provided context to answer.
- The user may ask the same information using different wording.
- If the context contains the answer, explain it naturally.
- Do not require exact keyword matching.
- If the information is partially available, answer using the available information.
- Only say "I couldn't find this information" if the context truly does not contain the answer.

Context:
{context}

Question:
{question}
"""


# ---------------- APP ----------------

app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- SCHEMAS ----------------


class AskRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str]



# ---------------- RAG FUNCTIONS ----------------


def retrieve_context(
    query: str,
    top_k: int = TOP_K
):

    query_embedding = (
        embedder.encode(query)
        .tolist()
    )


    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
        },
    ).execute()


    return result.data or []



def build_messages(
    question: str,
    chunks: list,
    history=None
):

    context = "\n\n".join(
        [
            f"""
Source:
{c['metadata'].get('source')}

Content:
{c['content']}
"""
            for c in chunks
        ]
    )


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


    # only add valid history messages
    if history:
        valid_history = [
            msg for msg in history
            if msg.get("role") and msg.get("content")
        ]

        messages.extend(valid_history)


    messages.append(
        {
            "role": "user",
            "content": f"""
Context:

{context}


Question:
{question}


Answer using only the context.
"""
        }
    )


    return messages



# ---------------- ROUTES ----------------


@app.get("/")
def health_check():

    return {
        "status": "ok",
        "message": "RAG API running"
    }



@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

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


    try:

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


    except Exception as e:

        print("GROQ ERROR:", e)

        return AskResponse(
            answer=f"Model service error: {str(e)}",
            sources=sources
        )


    return AskResponse(
        answer=answer,
        sources=sources
    )



# ---------------- START SERVER ----------------

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

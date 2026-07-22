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
from huggingface_hub import InferenceClient

from dotenv import load_dotenv

load_dotenv()


# ---------------- CONFIG ----------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = os.getenv(
    "MODEL_ID",
    "meta-llama/Llama-3.1-8B-Instruct"
)


if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_KEY environment variables"
    )


if not HF_TOKEN:
    raise RuntimeError(
        "Missing HF_TOKEN environment variable"
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


client = InferenceClient(
    api_key=HF_TOKEN
)


# ---------------- PROMPT ----------------

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions only using the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not in the context, say you do not know.
- Be concise and clear.
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

        print("HUGGINGFACE ERROR:", e)

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

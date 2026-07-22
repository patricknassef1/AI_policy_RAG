"""
app.py - FastAPI RAG API
"""

import os
import traceback
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from supabase import create_client
from groq import Groq

from dotenv import load_dotenv

load_dotenv()


# ================= CONFIG =================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_ID = os.getenv(
    "GROQ_MODEL_ID",
    "llama-3.3-70b-versatile"
)

TOP_K = 4

# Must match your Supabase vector dimension
EMBED_MODEL = "all-MiniLM-L6-v2"


# ================= ENV CHECK =================

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")

if not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")


# ================= CLIENTS =================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


embedder = SentenceTransformer(
    EMBED_MODEL
)


groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ================= PROMPT =================

SYSTEM_PROMPT = """
You are a helpful assistant answering questions based only on the provided documents.

Rules:
1. Use only the provided context.
2. The user may ask questions using different wording.
3. Match the meaning, not only exact keywords.
4. If the answer exists in the context, explain it clearly.
5. If information is partially available, provide the available information.
6. Do not invent information.
7. Only say "I couldn't find this information in the documents" when the context truly does not contain the answer.
"""


# ================= APP =================

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


# ================= SCHEMAS =================


class AskRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None



class AskResponse(BaseModel):
    answer: str
    sources: List[str]



# ================= EMBEDDING =================


def create_embedding(text: str):

    embedding = embedder.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()



# ================= RETRIEVAL =================


def retrieve_context(
    query: str,
    top_k: int = TOP_K
):

    query_embedding = create_embedding(query)

    try:

        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": top_k
            }
        ).execute()


        return response.data or []


    except Exception as e:

        print("SUPABASE ERROR")
        traceback.print_exc()

        raise e



# ================= MESSAGE BUILDER =================


def build_messages(
    question: str,
    chunks: list,
    history=None
):

    context_parts = []


    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {}
        ) or {}


        source = metadata.get(
            "source",
            "unknown"
        )


        content = chunk.get(
            "content",
            ""
        )


        context_parts.append(
            f"""
Source:
{source}

Content:
{content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


    # Add previous chat safely
    if history:

        for msg in history:

            if (
                isinstance(msg, dict)
                and msg.get("role")
                and msg.get("content")
            ):

                messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )


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



# ================= ROUTES =================


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
def ask(
    request: AskRequest
):

    try:

        # 1. Retrieve documents

        chunks = retrieve_context(
            request.question
        )


        if not chunks:

            return AskResponse(
                answer="I couldn't find this information in the documents.",
                sources=[]
            )


        # 2. Build prompt

        messages = build_messages(
            request.question,
            chunks,
            request.history
        )


        # 3. Extract sources

        sources = list(
            {
                (
                    c.get("metadata", {}) or {}
                ).get(
                    "source",
                    "unknown"
                )
                for c in chunks
            }
        )


        # 4. Call LLM

        response = groq_client.chat.completions.create(

            model=MODEL_ID,

            messages=messages,

            max_tokens=500,

            temperature=0.2
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


        print("APPLICATION ERROR:")
        traceback.print_exc()


        return AskResponse(

            answer=f"Server error: {str(e)}",

            sources=[]
        )



# ================= START =================


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

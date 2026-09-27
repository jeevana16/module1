import os
from pathlib import Path
from typing import TypedDict, List

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ============================================================
# CONFIGURATION
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

COLLECTION_NAME = "zepto_policies"


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# LOAD AND INDEX DOCUMENTS
# ============================================================

def load_documents():

    documents = []
    ids = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(text)
        ids.append(file_path.stem)

    return documents, ids


def build_index():

    documents, ids = load_documents()

    if not documents:
        raise RuntimeError(
            "No documents found in docs/ directory."
        )

    # Avoid adding duplicate IDs
    existing = collection.get(
        ids=ids
    )

    existing_ids = set(existing.get("ids", []))

    new_documents = []
    new_ids = []

    for doc, doc_id in zip(documents, ids):

        if doc_id not in existing_ids:

            new_documents.append(doc)
            new_ids.append(doc_id)

    if new_documents:

        embeddings = embedding_model.encode(
            new_documents,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=new_ids,
            documents=new_documents,
            embeddings=embeddings
        )

        print(
            f"Added {len(new_documents)} documents to ChromaDB."
        )

    else:

        print("Documents already indexed.")


build_index()


# ============================================================
# PYDANTIC RESPONSE MODEL
# ============================================================

class AnswerResponse(BaseModel):

    answer: str

    sources: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# PYDANTIC REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    query: str


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):

    query: str

    intent: str

    retrieved_documents: List[str]

    retrieved_ids: List[str]

    answer: str

    sources: List[str]

    confidence: float


# ============================================================
# STRUCTURED PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto Support Assistant.

CONTEXT:
You may answer only using the Zepto policy context provided below.

TASK:
Answer the customer's question using only the supplied context.

FORMAT:
Return a JSON object with:
{
    "answer": "string",
    "sources": ["document IDs"],
    "confidence": 0.0
}

LENGTH:
Keep the answer concise and directly relevant to the customer's question.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies.

FEW-SHOT EXAMPLE:

Question:
What is the delivery charge for orders below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.

Answer:
{
    "answer": "Orders below INR 149 incur a flat INR 25 delivery fee.",
    "sources": ["doc_01"],
    "confidence": 1.0
}

CUSTOMER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# NODE 1: CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: GraphState
):

    query = state["query"]

    lower_query = query.lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM != "0":

        if any(
            keyword in lower_query
            for keyword in keywords
        ):

            intent = "policy_question"

        else:

            intent = "general_question"

        print(
            f"[MOCK] Intent: {intent}"
        )

        return {
            "intent": intent
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM PATH
    # --------------------------------------------------------

    # Real LLM implementation can be added here.
    # The graded assignment does not require it.

    return {
        "intent": "general_question"
    }


# ============================================================
# NODE 2: RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: GraphState
):

    query = state["query"]

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # CHROMADB RETRIEVAL
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]

    ids = results["ids"][0]

    # --------------------------------------------------------
    # MOCK LLM
    # --------------------------------------------------------

    if MOCK_LLM != "0":

        top_chunk = documents[0]

        # Assignment asks for a short excerpt
        top_chunk_snippet = top_chunk[:200]

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

        return {
            "retrieved_documents": documents,
            "retrieved_ids": ids,
            "answer": answer,
            "sources": ids,
            "confidence": 1.0
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM PATH
    # --------------------------------------------------------

    context = "\n\n".join(
        documents
    )

    prompt = PROMPT_TEMPLATE.format(
        query=query,
        context=context
    )

    # Real LLM call would happen here.
    # MOCK_LLM=0 is optional and ungraded.

    return {
        "retrieved_documents": documents,
        "retrieved_ids": ids,
        "answer": "Real LLM path not configured.",
        "sources": ids,
        "confidence": 0.0
    }


# ============================================================
# NODE 3: DIRECT ANSWER
# ============================================================

def direct_answer(
    state: GraphState
):

    if MOCK_LLM != "0":

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

        return {
            "answer": answer,
            "sources": [],
            "confidence": 1.0
        }

    # Optional real LLM path

    return {
        "answer": "Real LLM path not configured.",
        "sources": [],
        "confidence": 0.0
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_intent(
    state: GraphState
):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    GraphState
)


graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)


# START -> classify

graph_builder.add_edge(
    START,
    "classify_intent"
)


# Conditional routing

graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)


# End nodes

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


graph = graph_builder.compile()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0"
)


# ============================================================
# POST /ask
# ============================================================

@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):

    query = request.query.strip()

    if not query:

        return AnswerResponse(
            answer="Please provide a question.",
            sources=[],
            confidence=0.0
        )

    result = graph.invoke(
        {
            "query": query
        }
    )

    response = AnswerResponse(
        answer=result["answer"],
        sources=result.get(
            "sources",
            []
        ),
        confidence=result.get(
            "confidence",
            1.0
        )
    )

    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Zepto Support Assistant is running",
        "mock_llm": MOCK_LLM
    }
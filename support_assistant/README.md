# Zepto Support Assistant

A RAG-based customer support assistant for answering Zepto policy-related questions using a local knowledge base.

## Features

- Loads policy documents from the `docs/` directory
- Splits documents into chunks
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores and retrieves documents using ChromaDB
- Uses LangGraph for intent-based routing
- Supports support/policy questions and general questions
- Uses a deterministic mock LLM, so no API key is required
- Returns structured Pydantic responses
- Provides a FastAPI REST API
- Includes Docker support

## Project Structure

```text
module3/
├── docs/
│   ├── doc_1.txt
│   ├── doc_2.txt
│   ├── doc_3.txt
│   ├── doc_4.txt
│   ├── doc_5.txt
│   ├── doc_6.txt
│   ├── doc_7.txt
│   └── doc_8.txt
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Architecture

```text
User Question
      |
      v
Intent Classification
      |
      +------------------+
      |                  |
   Support             General
      |                  |
      v                  v
ChromaDB Retrieval   Direct Answer
      |
      v
Relevant Documents
      |
      v
Structured Prompt
      |
      v
Mock LLM
      |
      v
Pydantic Response
```

## Technologies

- Python
- FastAPI
- Uvicorn
- LangGraph
- LangChain
- Sentence Transformers
- ChromaDB
- Pydantic

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts policy documents and user questions into vector embeddings for semantic retrieval.

## Knowledge Base

The `docs/` directory contains eight policy documents covering:

1. Delivery
2. Order cancellation
3. Refunds
4. Missing items
5. Damaged or defective items
6. Payments
7. Customer support
8. Product availability

## Installation

From the `module3` directory:

```powershell
py -m pip install --user -r requirements.txt
```

## Run the Application

Build/load the knowledge base:

```powershell
py main.py
```

Start the FastAPI server:

```powershell
py -m uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoint

### POST `/ask`

Example request:

```json
{
  "question": "How can I cancel my order?"
}
```

Example response:

```json
{
  "answer": "Order cancellation depends on the current order status and applicable cancellation rules.",
  "sources": ["doc_2.txt"],
  "confidence": 0.8
}
```

## Example Questions

Support question:

```text
How can I cancel my order?
```

General question:

```text
What is Python?
```

## Prompt Design

The structured prompt includes:

- Role
- Context
- Task
- Format
- Length
- Negative constraints
- Few-shot example

The assistant uses retrieved policy context for support questions and avoids inventing unsupported policy information.

## Testing

The application was tested locally using the FastAPI Swagger interface.

Example endpoint:

```text
POST /ask
```

Example request:

```json
{
  "question": "How can I cancel my order?"
}
```

The application successfully loaded the eight knowledge-base documents into ChromaDB and successfully started the FastAPI server.

## Docker

Build the Docker image:

```powershell
docker build -t zepto-support-assistant .
```

Run the container:

```powershell
docker run -p 8000:8000 zepto-support-assistant
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Notes

The project uses a deterministic mock LLM and does not require an external LLM API key.

The embedding model is downloaded automatically by Sentence Transformers on first use.
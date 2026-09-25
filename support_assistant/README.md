Support Assistant
=================

A small RAG service for Zepto. It embeds 8 policy documents, routes
each query through a LangGraph flow, validates the output with
Pydantic, and serves it over FastAPI.

The graded path runs fully offline. No LLM API key is needed when
MOCK_LLM is left at its default.

Install
-------

    pip install fastapi uvicorn langgraph langchain-core chromadb sentence-transformers pydantic

Run
---

    cd support_assistant
    python -m uvicorn main:app --host 127.0.0.1 --port 8000

Then:

    POST http://127.0.0.1:8000/ask
    Body: {"query": "..."}

Files
-----

docs/doc_01.txt ... docs/doc_08.txt   the 8 policy documents
main.py                               full pipeline and FastAPI app
Dockerfile                            container definition
requirements.txt                      dependencies

Pipeline
--------

Four stages, in order.

Ingestion
    load_documents() in main.py reads the 8 doc files and returns
    id and text for each one.

Embedding
    sentence-transformers loads all-MiniLM-L6-v2 locally. Each
    document is embedded once and stored in a ChromaDB collection
    called zepto_policies.

Retrieval
    Inside the retrieve_and_answer node, the query is embedded with
    the same model and ChromaDB returns the top 3 chunks by cosine
    similarity. This runs for real in both MOCK_LLM modes.

Generation
    This is the only stage that branches on MOCK_LLM. In mock mode
    the answer is a templated string built from the top retrieved
    chunk, and no LLM is called. In the optional MOCK_LLM=0 mode a
    real LLM would be called with the structured prompt template.

LangGraph
---------

Three nodes:

    classify_intent        picks policy_question or general_question
    retrieve_and_answer    embeds, retrieves, answers
    direct_answer          returns a fixed canned string

A conditional edge from classify_intent routes to one of the two
answer nodes. classify_intent uses a keyword list in mock mode and
does not call an LLM.

Pydantic output
---------------

Every response is an AskResponse with three fields:

    answer      string
    sources     list of doc IDs, empty for general questions
    confidence  float between 0 and 1

In mock mode sources comes from the retrieved chunks for policy
questions and is empty otherwise. Confidence is 1.0.

Prompt template
---------------

main.py has a STRUCTURED_PROMPT_TEMPLATE with all five parts:
ROLE, CONTEXT, TASK, FORMAT, LENGTH. It also includes a negative
constraint (do not use outside knowledge) and a few-shot example
("Is delivery free?"). The template is used only by the optional
real-LLM path.

Example calls
-------------

Both calls were run with MOCK_LLM at its default.

Policy question:

    POST /ask
    Body: {"query": "What is the delivery policy?"}

    {
      "answer": "based on the retrieved context Delivery Policy: Zepto delivers grocery...",
      "sources": ["doc_01", "doc_05", "doc_02"],
      "confidence": 1.0
    }

General question:

    POST /ask
    Body: {"query": "What is the capital of France?"}

    {
      "answer": "I can only answer questions about Zepto policies right now",
      "sources": [],
      "confidence": 1.0
    }

MOCK_LLM toggle
---------------

    MOCK_LLM = os.environ.get("MOCK_LLM", "1") == "1"

Default (unset or 1): mock mode. No LLM call, no API key, no network.
Classification uses keywords, the answer is templated from the top
chunk, and the Pydantic fields are set deterministically.

0: optional real-LLM mode. Only the generation step changes.
Retrieval still runs for real. This mode is ungraded and not required.

Docker
------

Build:

    cd support_assistant
    docker build -t zepto-support-assistant .

Run:

    docker run -p 7860:7860 zepto-support-assistant

The container serves POST /ask on port 7860. The Dockerfile is
included as the required local baseline. Hugging Face deployment was
not attempted.
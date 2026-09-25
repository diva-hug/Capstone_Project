# Module -3 Support Assistant:
print("Task 1: Load docs, embed with all-MiniLM-L6-v2, store in ChromaDB")

import os
import glob
from typing import List,TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph,END
from pydantic import BaseModel, Field
from fastapi import FastAPI

DOC_DIR =os.path.join(os.path.dirname(__file__),"docs")
COLLECTION_NAME = "zepto_policies"
MOCK_LLM =os.environ.get("MOCK_LLM","1")=="1"
# Task 2: Structured Prompt Template
# Role - Context - Task - Format - Length

STRUCTURED_PROMPT_TEMPLATE = """
ROLE:
You are Zepto's customer support assistant. Answer user questions
only about Zepto's delivery, returns, refunds, membership, tracking,
cancellation, gift cards, and support-hours policies.

CONTEXT:
{retrieved_context}

TASK:
Answer the user's question using only the information provided
in the context above. If the answer is not available in the
context, say that you do not have that information.

NEGATIVE CONSTRAINT:
Do not use outside knowledge. Do not invent or assume any policies,
prices, fees, or timelines that are not mentioned in the context.

FEW-SHOT EXAMPLE:

User question:
"Is delivery free?"

Retrieved context:
"Standard delivery is free on orders over INR 149.
Orders below this threshold incur a flat INR 25 delivery fee."

Answer:
"Standard delivery is free on orders over INR 149.
Orders below INR 149 incur a flat INR 25 delivery fee."

FORMAT:
Return only a short plain-text paragraph.
Do not use JSON, Markdown, or code.

LENGTH:
Maximum 3 sentences.

USER QUESTION:
{query}
"""
## Task 4: Pydantic output schema
class AskResponse(BaseModel):
    answer: str =Field(...,description="The final answer text.")
    sources :List[str]=Field(default_factory=list,description="Chunk/doc IDs used.")
    confidence:float=Field(...,ge=0.0,le=1.0,description="confidence 0-1")

class AskRequest(BaseModel):
    query:str =Field(...,description="user_query")


def load_documents()->List[dict]:
    """Load all doc_*.txt files from the docs folder."""
    files =sorted(glob.glob(os.path.join(DOC_DIR,"doc_*.txt")))
    docs=[]
    for path in files:
        with open(path,"r",encoding="utf-8")as f:
            text=f.read().strip()
        doc_id=os.path.basename(path).replace(".txt","")
        docs.append({"id":doc_id, "text":text})
    return docs

print("Loading documnts")
documents = load_documents()
print(f"Load documents of len {len(documents)}")

print("Load the embedding model (all-MiniLM-L6-v2)")
embedder =SentenceTransformer("all-MiniLM-L6-v2")

print("Building chroma collections")
chrom_client =chromadb.Client()
try:
    chrom_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection =chrom_client.create_collection(name=COLLECTION_NAME)

texts =[d["text"] for  d in documents]
ids=[d["id"] for d in documents]

embedding =embedder.encode(texts,show_progress_bar=False).tolist()

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embedding
)

print(f"ChormaDB collection {COLLECTION_NAME} built  with {collection.count()}chunks")

# check query collection 

test_query ="what is delivery  policy"
test_emb =embedder.encode([test_query]).tolist()
results =collection.query(query_embeddings=test_emb,n_results=1)

print(f"\nfirst check {test_query}")
print(f"\nTop match Documents id : {results['ids'][0][0]}")
print(f"\nTop match snippet: {results['documents'][0][0][:120]}")


print("Task 3: LangGraph StateGraph with 3 nodes and a conditional edge")

class GraphState(TypedDict):
    query:str
    intent:str
    retrieved:List[dict]
    answer:str
    sources:List[str]
    confidence:float

POLICY_KEYWORDS =[
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

def classify_intent(state: GraphState) ->GraphState:
    query=state["query"].lower()

    if any (kw in query for kw in POLICY_KEYWORDS):
        state["intent"] ="policy_question"
    else:
        state["intent"]="general_question"
    return state

def retrieve_and_answers(state:GraphState) ->GraphState:
    "Embedding Top 3 chunking "
    query = state["query"]

    # retrivel no need API 
    query_emb =embedder.encode([query]).tolist()
    results =collection.query(query_embeddings=query_emb,n_results=3)

    retrieved =[]
    if results and results.get("ids") and results["ids"][0]:
        for i , chunk_id in enumerate(results["ids"][0]):
            retrieved.append({
                "id":chunk_id,
                "text":results["documents"][0][i],
            })
    state["retrieved"]=retrieved

    if MOCK_LLM:
        if retrieved:
            snippet=retrieved[0]["text"][:200]
            state["answer"]=f"based on the retrived context{snippet}"
        else:
            state["answer"]="based on the retrived context no matching policy"
    else:
        if retrieved:
            snippet=retrieved[0]["text"][:200]
            state["answer"]=f"based on the retrived context:{snippet}"
        else:
            state["answer"]="based on the retrieved context no matching policy"
    return state


def direct_answer(state: GraphState) -> GraphState:
    state["answer"] = "I can only  answer questions about Zepto policies right now "
    state["retrieved"]=[]
    return state

def route_intent(state: GraphState) -> str:
    if state["intent"]=="policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# Buid the Graph 
graph_builder = StateGraph(GraphState)
graph_builder.add_node("classify_intent",classify_intent)
graph_builder.add_node("retrieve_and_answer",retrieve_and_answers)
graph_builder.add_node("direct_answer",direct_answer)

graph_builder.set_entry_point("classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

graph_builder.add_edge("retrieve_and_answer",END)
graph_builder.add_edge("direct_answer",END)

graph =graph_builder.compile()
print("\n Task 3 LangGraph compiled successfully")


# quick test
print("\n Test 1: policy question")
res1 =graph.invoke({"query":"what is the delivery policy?"})
print(f"Intent : {res1['intent']}")
print(f"Answer :{res1['answer'][:150]}")


print("\n Test 2 : general question")
res2 =graph.invoke({"query":"What is the capital of France?"})
print(f"Intent : {res2['intent']}")
print(f"Answer :{res2['answer'][:150]}")


def ask(query:str)->AskResponse:
    result=graph.invoke({"query":query})

    intent=result.get("intent","general_question")
    retrived =result.get("retrieved",[])
    answer_text =result.get("answer","")

    if intent =="policy_question":
        sources=[r["id"]for r  in retrived]
        confidence =1.0
    else:
        sources=[]
        confidence=1.0
    return AskResponse(
        answer=answer_text
        ,sources=sources,
        confidence=confidence,
    )

# Task4 test pydantic  validated output

print("\n--- Task 4: Pydantic-validated output ---")

resp1 =ask("what is  the delivery policy?")
print(f"answer :{resp1.answer[:120]}")
print(f"sources: {resp1.sources}")
print(f"confidence:{resp1.confidence}")


resp2 =ask("what is the capital of india?")
print(f"answer :{resp2.answer[:120]}")
print(f"sources: {resp2.sources}")
print(f"confidence:{resp2.confidence}")

# TASK 5 
print( "Task 5: FastAPI wrapper")

app=FastAPI(title="Zepto support Assistant")

@app.get("/")
def root():
    return {"status":"ok","serivice":"Zepto support Assistant"}

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request:AskRequest)->AskResponse:
    return ask(request.query)
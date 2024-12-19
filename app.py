from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from setup_rag import load_model_and_index, query_rag_framework

# Load model, retriever, and data
model_path = "./fine_tuned_model"
index_file = "faiss_index"
dump_file = "parsed_wikipedia.jsonl"

model, tokenizer, index, passages, vectorizer = load_model_and_index(model_path, index_file, dump_file)

# Define API app
app = FastAPI(title="RAG API", description="An API for Retrieval-Augmented Generation", version="1.0.0")

# Input schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Number of passages to retrieve

# API Routes
@app.post("/generate/")
def generate_answer(request: QueryRequest):
    """
    Generate an answer based on the query using the RAG framework.
    """
    try:
        response = query_rag_framework(
            query=request.query,
            index=index,
            vectorizer=vectorizer,
            passages=passages,
            tokenizer=tokenizer,
            model=model,
            top_k=request.top_k,
        )
        return {"query": request.query, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the RAG API. Use /generate endpoint to get answers."}

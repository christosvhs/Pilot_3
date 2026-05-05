from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import RAGPipeline

app = FastAPI(title="RAG Pipeline API")
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    dialog_history: list[str]


class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_documents: dict


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.dialog_history:
        raise HTTPException(status_code=400, detail="dialog_history must not be empty")
    result = pipeline.run(request.dialog_history)
    return QueryResponse(
        query=result["query"],
        response=result["response"],
        retrieved_documents=result["retrieved_documents"],
    )

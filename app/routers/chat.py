from fastapi import APIRouter, Query, HTTPException
from app.core.chat_model import ChatModel

router = APIRouter()
chat_model = ChatModel() #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@router.get("/chat")
async def chat(
    question: str = Query(..., description="Type your question here", max_length=256),
):
    try:
        response = chat_model.generate_response(question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "ok"}
from fastapi import APIRouter, Query, HTTPException, WebSocket, Request, Form
from fastapi.responses import HTMLResponse
from app.core.chat_model import ChatModel
from fastapi.templating import Jinja2Templates

router = APIRouter()
chat_model = ChatModel() #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def chat(
    request: Request,
    question: str = Query(None, description="Type your question here", max_length=256),
):
    context = {"request": request, "question": question, "answer": None}
    
    if question:
        try:
            response = chat_model.generate_response(question)
            context["answer"] = response
        except Exception as e:
            context["error"] = str(e)
    
    return templates.TemplateResponse("chat.html", context)

@router.post("/", response_class=HTMLResponse)
async def chat_submit(
    request: Request,
    question: str = Form(...)
):
    try:
        response = chat_model.generate_response(question)
        context = {
            "request": request,
            "question": question,
            "answer": response
        }
    except Exception as e:
        context = {
            "request": request,
            "question": question,
            "error": str(e)
        }
    
    return templates.TemplateResponse("chat.html", context)

@router.get("/health")
async def health():
    return {"status": "ok"}
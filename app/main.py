from fastapi import FastAPI, WebSocket
from app.core.config import settings
from app.routes import classification
from app.routes import browser_use
from app.routes import action_runner
from app.routes import summarizer
from app.routes import human
#from app.routes.speech_to_text import router as speech_router

app = FastAPI(
    title=settings.APP_NAME,
    description="API for classifying text using LangGraph and OpenAI",
    version="1.0.0"
)

# Include routers
app.include_router(classification.router)
app.include_router(browser_use.router)
app.include_router(action_runner.router)
app.include_router(summarizer.router)
app.include_router(human.router)

# Add WebSocket endpoint
#@app.websocket("/ws/speech-to-text")
#async def websocket_endpoint(websocket: WebSocket):
 #   await speech_router.handle_websocket(websocket)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the LLM Text Classifier API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    } 
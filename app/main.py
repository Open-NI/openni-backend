from fastapi import FastAPI
from app.core.config import settings
from app.routes import classification

app = FastAPI(
    title=settings.APP_NAME,
    description="API for classifying text using LangGraph and OpenAI",
    version="1.0.0"
)

# Include routers
app.include_router(classification.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the LLM Text Classifier API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    } 
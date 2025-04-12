from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.routes.human import text_to_speech
from app.services.langgraph_service import LangGraphService
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(prefix="/api/v1", tags=["summarizer"])

class SummarizerRequest(BaseModel):
    """Request model for text summarization."""
    text: str
    max_tts_words: Optional[int] = 50  # Target word count for TTS summary
    voice: Optional[str] = 'af_heart'  # Default voice for TTS

class SummarizerResponse(BaseModel):
    """Response model for text summarization."""
    tts_summary: str  # Concise summary for text-to-speech (10-20 seconds)
    tts_audio: Optional[bytes] = None  # Audio data for TTS summary
    detailed_summary: str  # Broader summary for HTML display
    error: Optional[str] = None

def get_langgraph_service():
    """Dependency injection for LangGraphService."""
    return LangGraphService()

@router.post("/summarize", response_model=SummarizerResponse)
async def summarize_text(
    request: SummarizerRequest,
    langgraph_service: LangGraphService = Depends(get_langgraph_service)
):
    """
    Generate two types of summaries from the input text:
    1. A concise summary suitable for text-to-speech (10-20 seconds)
    2. A more detailed summary for HTML display
    
    Args:
        request: The summarization request containing the text to summarize
        langgraph_service: The LangGraph service for LLM operations
        
    Returns:
        SummarizerResponse: Contains both types of summaries
        
    Raises:
        HTTPException: If there's an error during summarization
    """
    try:
        logger.debug(f"Received summarization request with {len(request.text)} characters")
        
        # Generate TTS-friendly summary
        tts_prompt = f"""Summarize the following text in a very concise way that would take 10-20 seconds to speak out loud.
        The summary should be natural to speak and easy to follow when heard.
        Target word count: {request.max_tts_words}
        
        Text to summarize:
        {request.text}
        """
        
        tts_messages = [
            {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries optimized for text-to-speech."},
            {"role": "user", "content": tts_prompt}
        ]
        
        # Generate detailed summary
        detailed_prompt = f"""Create a comprehensive summary of the following text.
        The summary should be well-structured and include the main points and important details.
        Use clear formatting and structure to make it easy to read when displayed in HTML.
        
        Text to summarize:
        {request.text}
        """
        
        detailed_messages = [
            {"role": "system", "content": "You are a helpful assistant that creates well-structured, detailed summaries."},
            {"role": "user", "content": detailed_prompt}
        ]
        
        # Get summaries from LLM
        tts_response = await langgraph_service.llm.ainvoke(tts_messages)
        detailed_response = await langgraph_service.llm.ainvoke(detailed_messages)
        
        tts_summary = tts_response.content.strip()
        detailed_summary = detailed_response.content.strip()
        
        logger.debug(f"Generated TTS summary with {len(tts_summary.split())} words")
        logger.debug(f"Generated detailed summary with {len(detailed_summary.split())} words")

        voice_audio = text_to_speech.text_to_speech(
            text=tts_summary,
            voice=request.voice
        )
        
        return SummarizerResponse(
            tts_summary=tts_summary,
            detailed_summary=detailed_summary,
            tts_audio=voice_audio,
        )
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summaries: {str(e)}"
        ) 
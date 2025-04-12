import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, SystemPrompt
from typing import Optional
from fastapi.responses import StreamingResponse
import io
from app.speech_to_text import SpeechToText
from app.text_to_speech import TextToSpeech

router = APIRouter(prefix="/api/v1/human", tags=["human"])

# Initialize services
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()

# Define request and response models
class SpeechToTextResponse(BaseModel):
    text: str

class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = 'af_heart'

@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Transcribe the audio
        text = speech_to_text.transcribe(temp_path, language)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return SpeechToTextResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-to-speech")
async def convert_text_to_speech(request: TextToSpeechRequest):
    try:
        # Convert text to speech
        audio_data = text_to_speech.text_to_speech(request.text, request.voice)
        
        # Return the audio file
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

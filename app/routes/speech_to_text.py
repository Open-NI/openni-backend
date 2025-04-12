from fastapi import WebSocket, WebSocketDisconnect
from app.services.speech_to_text import SpeechToTextService
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextRouter:
    def __init__(self):
        self.service = SpeechToTextService()
        
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("New WebSocket connection established")
        
        try:
            while True:
                # Receive audio data from the client
                data = await websocket.receive_bytes()
                logger.info(f"Received audio data: {len(data)} bytes")
                
                # Process the audio data with Whisper
                transcription = await self.service.transcribe_audio(data)
                
                # Send the transcription back to the client
                response = {
                    "text": transcription,
                    "status": "success" if transcription else "error"
                }
                await websocket.send_json(response)
                logger.info(f"Sent transcription: {transcription[:50]}...")
                
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {str(e)}")
            await websocket.close()

router = SpeechToTextRouter() 
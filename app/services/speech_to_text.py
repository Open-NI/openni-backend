import torch
import whisper
import numpy as np
import io
from typing import Optional
import os

class SpeechToTextService:
    def __init__(self):
        # Check for Metal (MPS) support on macOS
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Path to local model
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "whisper")
        os.makedirs(model_path, exist_ok=True)
        
        # Load local model
        self.model = whisper.load_model(
            "base",
            device=self.device,
            download_root=model_path
        )
        
    async def transcribe_audio(self, audio_data: bytes) -> str:
        try:
            # Convert bytes to numpy array and ensure it's float32
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Ensure the audio is in the correct format (16kHz, mono)
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)  # Convert to mono if stereo
            
            # Normalize audio
            audio_np = audio_np / np.max(np.abs(audio_np))
            
            # Transcribe the audio
            result = self.model.transcribe(
                audio_np,
                language="en",  # Specify language if known
                task="transcribe"
            )
            
            return result["text"].strip()
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return "" 
import numpy as np
import io
from typing import Optional

class TextToSpeech:
    def __init__(self):
        self.pipeline = None

    def text_to_speech(self, text: str, voice: Optional[str] = 'af_heart') -> bytes:
        """
        Convert text to speech using Kokoro-82M model
        
        Args:
            text (str): Input text to convert to speech
            voice (str, optional): Voice to use. Defaults to 'af_heart'
            
        Returns:
            bytes: WAV audio data
        """

        if not self.pipeline:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='a')

        # Generate audio
        generator = self.pipeline(text, voice=voice)
        
        # Collect all audio chunks
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)
            
        # Concatenate all audio chunks
        full_audio = np.concatenate(audio_chunks)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        import soundfile as sf
        sf.write(buffer, full_audio, 24000, format='WAV')
        buffer.seek(0)
        
        return buffer.getvalue() 
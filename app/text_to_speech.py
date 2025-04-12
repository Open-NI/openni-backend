import numpy as np
import io
from typing import Optional
import uuid
import os

class TextToSpeech:
    def __init__(self):
        self.pipeline = None

    def text_to_speech(self, text: str, voice: Optional[str] = 'af_heart') -> Optional[bytes]:
        """
        Convert text to speech using Kokoro-82M model

        Args:
            text (str): Input text to convert to speech
            voice (str, optional): Voice to use. Defaults to 'af_heart'

        Returns:
            bytes: WAV audio data
        """

        print(f'Converting text to speech (voice: {voice}): "{text}"')

        if not self.pipeline:
            try:
                from kokoro import KPipeline
                self.pipeline = KPipeline(lang_code='a')
            except ImportError:
                print("Kokoro library not found. Please install it.")
                return None

        try:
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
            filename = f"{uuid.uuid4()}.wav"
            self.save_to_file(buffer.getvalue(), directory='audio', filename=filename)

            print(f"TTS audio saved to {filename}")

            return filename
        except Exception as e:
            print(f"Error in text_to_speech: {str(e)}")
            return None

    def save_to_file(self, audio_data: bytes, directory: str = '.', filename: Optional[str] = None) -> str:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        """
        Save audio data to a WAV file with UUID filename

        Args:
            audio_data (bytes): Audio data in WAV format
            directory (str): Directory to save the file
            filename (Optional[str]): Filename to use (if None, uses UUID)

        Returns:
            str: Path to the saved file
        """
        if not filename:
            filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(directory, filename)

        with open(file_path, 'wb') as f:
            f.write(audio_data)

        return file_path

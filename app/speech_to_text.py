from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from typing import Optional
import platform
import requests

class SpeechToText:
    def __init__(self):
        # Check if we're on macOS and Metal is available
        is_mac = platform.system() == "Darwin"
        has_metal = torch.backends.mps.is_available() if is_mac else False
        
        if has_metal:
            self.device = "mps"  # Metal Performance Shaders
            self.torch_dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
        
        self.model_id = "openai/whisper-large-v3-turbo"
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file to text using Whisper model.
        
        Args:
            audio_path (str): Path to the audio file
            language (Optional[str]): Language code if known (e.g., "en", "fr")
            
        Returns:
            str: Transcribed text
        """
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            
        result = self.pipe(audio_path, generate_kwargs=generate_kwargs)
        print(result)

        # API is on localhost:8000/api/v1/classify
        api_url = "http://localhost:8000/api/v1/action-runner/begin"
        response = requests.post(api_url, json={
            "user": "Janez Novak",
            "request_message": result["text"],
        })

        if response.status_code == 200:
            print("Response from API:", response.json())
        else:
            print(f"Failed to send data to API. Status code: {response.status_code}, Response: {response.text}")

        response_data = response.json()

        if "task_id" in response_data:
            task_id = response_data["task_id"]
            print(f"Received task ID: {task_id}")
        else:
            print("No task ID found in the result.")

        return result["text"], response_data

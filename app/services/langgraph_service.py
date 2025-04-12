from openai import OpenAI
from app.core.config import settings
from app.models.classification import ClassificationLabel
from fastapi import HTTPException

class LangGraphService:
    """Service for handling LangGraph and OpenAI operations."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.MODEL_NAME
    
    def classify_text(self, text: str) -> tuple[ClassificationLabel, str | None]:
        """
        Classify the input text using OpenAI's GPT model and generate response if needed.
        
        Args:
            text: The text to classify
            
        Returns:
            tuple[ClassificationLabel, str | None]: The classification result and optional response
            
        Raises:
            HTTPException: If the classification is invalid
        """
        system_prompt = """You are a classifier that determines how to handle user requests. 
        Classify the request into one of these categories:
        - "browser_use": If the request needs to extract information from the web
        - "normal_response": If the request can be answered using general knowledge
        - "api_actions": If the request requires specific API actions
        
        Respond with ONLY the classification label, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate the classification
        if classification not in ["browser_use", "normal_response", "api_actions"]:
            raise HTTPException(status_code=500, detail="Invalid classification from LLM")
        
        # Generate response for normal_response classification
        llm_response = None
        if classification == "normal_response":
            llm_response = self._generate_response(text)
        
        return classification, llm_response
    
    def _generate_response(self, text: str) -> str:
        """
        Generate a response for the input text.
        
        Args:
            text: The input text
            
        Returns:
            str: The generated response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide a clear and concise response to the user's request."},
                {"role": "user", "content": text}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip() 
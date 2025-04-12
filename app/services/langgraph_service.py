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
    
    def classify_text(self, text: str) -> ClassificationLabel:
        """
        Classify the input text using OpenAI's GPT model.
        
        Args:
            text: The text to classify
            
        Returns:
            ClassificationLabel: The classification result
            
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
        
        return classification 
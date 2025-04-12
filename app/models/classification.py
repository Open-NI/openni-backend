from pydantic import BaseModel
from typing import Literal, Optional

# Define the classification labels
ClassificationLabel = Literal["browser_use", "normal_response", "api_actions"]

class ClassificationRequest(BaseModel):
    """Request model for text classification."""
    text: str

class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    classification: ClassificationLabel
    response: Optional[str] = None
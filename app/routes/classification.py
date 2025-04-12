from fastapi import APIRouter, Depends, HTTPException
from app.models.classification import ClassificationRequest, ClassificationResponse
from app.services.langgraph_service import LangGraphService

router = APIRouter(prefix="/api/v1", tags=["classification"])

def get_langgraph_service():
    """Dependency injection for LangGraphService."""
    return LangGraphService()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_request(
    request: ClassificationRequest,
    langgraph_service: LangGraphService = Depends(get_langgraph_service)
):
    """
    Classify the input text and provide an explanation.
    
    Args:
        request: The classification request
        langgraph_service: The LangGraph service
        
    Returns:
        ClassificationResponse: The classification result with explanation
        
    Raises:
        HTTPException: If there's an error during classification
    """
    try:
        classification = langgraph_service.classify_text(request.text)
        explanation = langgraph_service.generate_explanation(request.text, classification)
        
        return ClassificationResponse(
            classification=classification,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
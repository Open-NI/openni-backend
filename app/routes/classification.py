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
    Classify the input text.
    
    Args:
        request: The classification request
        langgraph_service: The LangGraph service
        
    Returns:
        ClassificationResponse: The classification result
        
    Raises:
        HTTPException: If there's an error during classification
    """
    try:
        classification = langgraph_service.classify_text(request.text)
        return ClassificationResponse(
            classification=classification
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
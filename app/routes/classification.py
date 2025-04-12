from fastapi import APIRouter, Depends, HTTPException
from app.models.classification import ClassificationRequest, ClassificationResponse
from app.services.langgraph_service import LangGraphService
from app.services.browser_service import BrowserService
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(prefix="/api/v1")

# Define dependencies
def get_langgraph_service():
    return LangGraphService()

def get_browser_service():
    return BrowserService()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_request(
    request: ClassificationRequest,
    langgraph_service: LangGraphService = Depends(get_langgraph_service)
):
    """
    Classify the input text and return a response.
    If the classification is 'browser_use', the response will include the enhanced search query.
    """
    try:
        logger.debug(f"Received classification request: {request.text}")
        
        # Process the text through the LangGraph workflow
        result = await langgraph_service.process_text(request.text)
        logger.debug(f"LangGraph result: {result}")
        
        # Extract the components
        classification = result.get("classification", "normal_response")
        response = result.get("response", "")
        browser_input = result.get("browser_input", "")
        
        # Return the classification response
        return ClassificationResponse(
            classification=classification,
            response=response,
            browser_input=browser_input
        )
        
    except Exception as e:
        logger.error(f"Error in classification request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 
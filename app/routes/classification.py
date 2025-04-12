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
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
    browser_service: BrowserService = Depends(get_browser_service)
):
    """
    Classify the input text and return a response.
    If the classification is 'browser_use', the response will include search results.
    """
    try:
        logger.debug(f"Received classification request: {request.text}")
        
        # Get classification from LangGraph
        classification, response, browser_input = await langgraph_service.classify_text(request.text)
        logger.debug(f"Classification: {classification}, Response: {response}, Browser Input: {browser_input}")
        
        # If classification is browser_use, get search results
        if classification == "browser_use" and browser_input:
            try:
                logger.debug(f"Running browser search for: {browser_input}")
                browser_result = await browser_service.run_browser(browser_input)
                logger.debug(f"Browser result: {browser_result}")
                
                # Return the browser result directly
                return ClassificationResponse(
                    classification=classification,
                    response=browser_result["result"],
                    browser_input=browser_input
                )
            except Exception as e:
                logger.error(f"Error in browser operation: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Browser operation failed: {str(e)}")
        
        # For non-browser classifications, return the normal response
        return ClassificationResponse(
            classification=classification,
            response=response,
            browser_input=browser_input
        )
        
    except Exception as e:
        logger.error(f"Error in classification request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 
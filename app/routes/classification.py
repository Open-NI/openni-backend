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
        
        # Process the text through the LangGraph workflow
        result = await langgraph_service.process_text(request.text)
        logger.debug(f"LangGraph result: {result}")

        print(result)
        
        # Extract the components
        classification = result.get("classification", "normal_response")
        response = result.get("response", "")
        browser_input = result.get("browser_input", "")
        browser_result = result.get("browser_result", None)
        
        # If we have a browser result, use it as the response
        if browser_result:
            print(browser_result)
            logger.debug(f"Using browser result as response")
            # Check if browser_result is a dictionary with a "result" key
            if isinstance(browser_result, dict) and "result" in browser_result:
                return ClassificationResponse(
                    classification=classification,
                    response=browser_result["result"],
                    browser_input=browser_input
                )
            # If it's just a string, use it directly
            elif isinstance(browser_result, str):
                return ClassificationResponse(
                    classification=classification,
                    response=browser_result,
                    browser_input=browser_input
                )
            # Otherwise, convert to string
            else:
                return ClassificationResponse(
                    classification=classification,
                    response=str(browser_result),
                    browser_input=browser_input
                )
        
        # For non-browser classifications, return the normal response
        return ClassificationResponse(
            classification=classification,
            response=response,
            browser_input=browser_input
        )
        
    except Exception as e:
        logger.error(f"Error in classification request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 
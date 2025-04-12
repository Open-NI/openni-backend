from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.action_runner import ActionRunnerBeginRequest, ActionRunnerStatusResponse, ActionRunnerBeginResponse
from app.models.classification import ClassificationRequest, ClassificationResponse
from app.services.langgraph_service import LangGraphService
from app.services.mongodb_service import mongodb_service
from app.routes.browser_use import handle_browser_use, BrowserUseRequest
from bson import ObjectId
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/action-runner", tags=["action-runner"])

def get_langgraph_service():
    """Dependency injection for LangGraphService."""
    return LangGraphService()

async def run_browser_task(action_id: str, request_message: str):
    """Background task to handle browser use requests."""
    try:
        logger.info(f"Starting browser task for action {action_id}")
        
        # Update status to indicate browser task is starting
        await mongodb_service.update_action_status(
            action_id=action_id,
            status="browser_task_started",
            explanation="Starting browser automation..."
        )
        
        browser_response = await handle_browser_use(BrowserUseRequest(prompt=request_message))
        
        logger.info(f"Browser task completed for action {action_id}")
        await mongodb_service.update_action_status(
            action_id=action_id,
            status="completed",
            result=browser_response.response
        )
    except Exception as e:
        logger.error(f"Error in browser task for action {action_id}: {str(e)}")
        await mongodb_service.update_action_status(
            action_id=action_id,
            status="failed",
            error_message=str(e)
        )

@router.post("/begin", response_model=ActionRunnerBeginResponse)
async def begin_request(
    request: ActionRunnerBeginRequest,
    background_tasks: BackgroundTasks,
    langgraph_service: LangGraphService = Depends(get_langgraph_service)
):
    """
    Begin the action runner process with the provided request.

    Args:
        request: The action runner request
        background_tasks: FastAPI background tasks
        langgraph_service: The LangGraph service
        
    Returns:
        ActionRunnerBeginResponse: The response with action ID and initial status
        
    Raises:
        HTTPException: If there's an error during the process
    """
    try:
        logger.info(f"Starting action for user {request.user}")
        
        # Get classification and response
        classification, response = langgraph_service.classify_text(request.request_message)
        logger.info(f"Classification result: {classification}")
        
        # Create initial action record in MongoDB with all classification details
        action_data = {
            "user": request.user,
            "request_message": request.request_message,
            "status": "running",
            "classification": classification,
            "classification_details": {
                "type": classification,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            },
            "explanation": "Processing request..."
        }
        
        action_id = await mongodb_service.create_action(action_data)
        logger.info(f"Created action with ID: {action_id}")
        
        # If it's a normal response, we can complete it immediately
        if classification == "normal_response" and response:
            logger.info(f"Completing normal response for action {action_id}")
            await mongodb_service.update_action_status(
                action_id=action_id,
                status="completed",
                result=response,
                explanation="Request completed with normal response"
            )
            action_data["status"] = "completed"
            action_data["result"] = response
        
        # If it requires browser use, start a background task
        elif classification == "browser_use":
            logger.info(f"Starting browser task for action {action_id}")
            background_tasks.add_task(run_browser_task, action_id, request.request_message)
        
        return ActionRunnerBeginResponse(
            classification=action_data["classification"],
            status=action_data["status"],
            explanation=action_data["explanation"],
            action_id=action_id
        )
    except Exception as e:
        logger.error(f"Error in begin_request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{action_id}", response_model=ActionRunnerStatusResponse)
async def get_status(action_id: str):
    """
    Get the status of the action runner process.
    
    Args:
        action_id: The ID of the action runner process
        
    Returns:
        ActionRunnerStatusResponse: The status of the action runner process
        
    Raises:
        HTTPException: If there's an error during status retrieval
    """
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(action_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid action ID format")
            
        action = await mongodb_service.get_action(object_id)
        if not action:
            raise HTTPException(status_code=404, detail="Action not found")
            
        return ActionRunnerStatusResponse(
            status=action["status"],
            result=action.get("result"),
            error_message=action.get("error_message")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

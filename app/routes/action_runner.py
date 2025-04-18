from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.action_runner import (
    ActionRunnerBeginRequest,
    ActionRunnerStatusResponse,
    ActionRunnerBeginResponse,
)
from app.models.classification import ClassificationRequest, ClassificationResponse
from app.services.langgraph_service import LangGraphService, API_ACTION_HANDLERS, CONVERSATION_HISTORY
from app.services.mongodb_service import mongodb_service
from app.services.browser_service import BrowserService
from app.routes.human import text_to_speech
from bson import ObjectId
import logging
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/action-runner", tags=["action-runner"])


def get_langgraph_service():
    """Dependency injection for LangGraphService."""
    return LangGraphService()


def get_browser_service():
    """Dependency injection for BrowserService."""
    return BrowserService()


def render_text_to_speech(text, voice):
    audio_file_path = text_to_speech.text_to_speech(text, voice)
    return audio_file_path


@router.post("/begin", response_model=ActionRunnerBeginResponse)
async def begin_request(
    request: ActionRunnerBeginRequest,
    background_tasks: BackgroundTasks,
    langgraph_service: LangGraphService = Depends(get_langgraph_service),
    browser_service: BrowserService = Depends(get_browser_service),
):
    """
    Begin the action runner process with the provided request.

    Args:
        request: The action runner request
        background_tasks: FastAPI background tasks
        langgraph_service: The LangGraph service
        browser_service: The Browser service

    Returns:
        ActionRunnerBeginResponse: The response with action ID and initial status

    Raises:
        HTTPException: If there's an error during the process
    """
    try:
        logger.info(f"Starting action for user {request.user}")

        # Process the request using LangGraph service with voice personalization
        result = await langgraph_service.process_text(
            request.request_message, request.voice
        )
        classification = result.get("classification", "normal_response")
        response = result.get("response", "")
        browser_input = result.get("browser_input", "")
        api_action = result.get("api_action")
        api_params = result.get("api_params")

        logger.info(f"Classification result: {classification}")

        # Create initial action record in MongoDB with all classification details
        action_data = {
            "user": request.user,
            "request_message": request.request_message,
            "voice": request.voice,
            "status": "running",
            "classification": classification,
            "classification_details": {
                "type": classification,
                "response": response,
                "browser_input": browser_input,
                "api_action": api_action,
                "api_params": api_params,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "explanation": "Processing request...",
        }

        action_id = await mongodb_service.create_action(action_data)
        logger.info(f"Created action with ID: {action_id}")

        # Handle different classification types
        if classification == "normal_response":
            logger.info(f"Completing normal response for action {action_id}")
            # Generate a response if not already provided
            if not response:
                response = await langgraph_service._generate_response(
                    request.request_message
                )

            print(f"Response: {response}")

            audio_data = render_text_to_speech(response, request.voice)
            action_data["tts_audio_base64"] = audio_data

            # tts_audio_base64 = base64.b64encode(tts_audio).decode('utf-8') if tts_audio else None

            await mongodb_service.update_action_status(
                action_id=action_id,
                status="completed",
                result=response,
                explanation="Request completed with normal response",
                tts_audio_base64=audio_data,
            )

            action_data["status"] = "completed"
            action_data["result"] = response

        elif classification == "browser_use":
            logger.info(f"Starting browser task for action {action_id}")
            try:
                # Update status to indicate browser task is starting
                await mongodb_service.update_action_status(
                    action_id=action_id,
                    status="browser_task_started",
                    explanation="Starting browser automation...",
                )

                # Run browser task directly
                browser_result = await browser_service.run_browser(
                    browser_input or request.request_message, action_id
                )
                print(f"Browser result: {browser_result}")
                
                # Extract the result from the browser response
                result_text = browser_result.get("result", "")
                
                if not result_text and isinstance(browser_result, dict):
                    # Try to extract from different possible locations in the response
                    if "extracted_content" in browser_result:
                        result_text = browser_result["extracted_content"]
                    elif "final_result" in browser_result:
                        result_text = browser_result["final_result"]
                    elif "content" in browser_result:
                        result_text = browser_result["content"]

                # If we still don't have a result, use a fallback
                if not result_text:
                    result_text = f"Browser task completed but no specific result was returned: {browser_input or request.request_message}"

                audio_data = render_text_to_speech(result_text, request.voice)
                action_data["tts_audio_base64"] = audio_data

                # Add the result to the global conversation history
                CONVERSATION_HISTORY.append({"role": "assistant", "content": result_text})
                
                # Update status with browser result
                await mongodb_service.update_action_status(
                    action_id=action_id,
                    status="completed",
                    result=result_text,
                    explanation="Browser task completed successfully",
                    tts_audio_base64=audio_data,
                )
                action_data["status"] = "completed"
                action_data["result"] = result_text

            except Exception as e:
                logger.error(f"Error in browser task for action {action_id}: {str(e)}")
                await mongodb_service.update_action_status(
                    action_id=action_id,
                    status="failed",
                    error_message=str(e),
                    explanation="Browser task failed",
                )
                action_data["status"] = "failed"
                action_data["error_message"] = str(e)

        elif classification == "api_actions":
            logger.info(f"Executing API action for action {action_id}")
            try:
                # Execute the API action
                if api_action and api_action in API_ACTION_HANDLERS:
                    if api_action == "MEME_CAPTION":
                        api_params = { "text": request.request_message }

                    print(f"API action: {api_action} with params: {api_params}")
                    action_result = API_ACTION_HANDLERS[api_action](
                        api_params
                    )
                    result_text = str(action_result)

                    print(f"Result text: {result_text}")

                    audio_data = render_text_to_speech(response, request.voice)
                    action_data["tts_audio_base64"] = audio_data

                    await mongodb_service.update_action_status(
                        action_id=action_id,
                        status="completed",
                        result=result_text,
                        explanation=f"API action {api_action} executed successfully",
                        tts_audio_base64=audio_data,
                    )
                    action_data["status"] = "completed"
                    action_data["result"] = result_text
                else:
                    raise ValueError(f"Unknown API action: {api_action}")
            except Exception as e:
                logger.error(f"Error executing API action: {str(e)}")
                await mongodb_service.update_action_status(
                    action_id=action_id,
                    status="failed",
                    error_message=str(e),
                    explanation=f"Failed to execute API action {api_action}",
                )
                action_data["status"] = "failed"
                action_data["error_message"] = str(e)

        # Return the response with the appropriate result
        return ActionRunnerBeginResponse(
            classification=action_data["classification"],
            status=action_data["status"],
            response=action_data.get("result"),  # Include the result in the response
            explanation=action_data["explanation"],
            action_id=action_id,
            tts_audio_base64=action_data.get("tts_audio_base64"),
        )
    except Exception as e:
        logger.error(f"Error in begin_request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{action_id}", response_model=ActionRunnerStatusResponse)
async def get_status(action_id: str, return_screenshot: bool = False):
    """
    Get the status of the action runner process.

    Args:
        action_id: The ID of the action runner process
        return_screenshot: Whether to return the screenshot (default: False)

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
            error_message=action.get("error_message"),
            tts_audio_base64=action.get("tts_audio_base64"),
            screenshot=action.get("screenshot") if return_screenshot else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

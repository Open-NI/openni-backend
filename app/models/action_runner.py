from pydantic import BaseModel
from typing import Literal, Optional

# Define all possible status values
ActionStatus = Literal["running", "completed", "failed", "browser_task_started"]

class ActionRunnerBeginRequest(BaseModel):
    """Request model for action runner begin."""
    user: str
    request_message: str
    voice: str

class ActionRunnerBeginResponse(BaseModel):
    """Response model for action runner begin."""
    classification: Literal["browser_use", "normal_response", "api_actions"]
    status: ActionStatus
    response: Optional[str] = None
    explanation: str
    action_id: str


class ActionRunnerStatusResponse(BaseModel):
    """Response model for action runner status."""
    status: ActionStatus
    result: Optional[str] = None
    error_message: Optional[str] = None

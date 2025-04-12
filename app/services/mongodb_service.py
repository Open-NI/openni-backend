from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, Union, Dict, Any, List
import os
from dotenv import load_dotenv
from bson import ObjectId
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBService:
    def __init__(self):
        self.client = AsyncIOMotorClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client.action_runner
        self.actions = self.db.actions

    async def create_action(self, action_data: dict) -> str:
        """Create a new action in the database."""
        result = await self.actions.insert_one({
            **action_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        return str(result.inserted_id)

    async def get_action(self, action_id: Union[str, ObjectId]) -> Optional[dict]:
        """Get an action by its ID."""
        # Convert string ID to ObjectId if needed
        if isinstance(action_id, str):
            try:
                action_id = ObjectId(action_id)
            except Exception:
                return None
                
        action = await self.actions.find_one({"_id": action_id})
        if action:
            action["_id"] = str(action["_id"])
        return action

    async def update_action_status(
        self,
        action_id: str,
        status: str,
        result: Optional[str] = None,
        error_message: Optional[str] = None,
        explanation: Optional[str] = None,
        model_thoughts: Optional[List[str]] = None
    ) -> bool:
        """
        Update the status of an action in the database.
        
        Args:
            action_id: The ID of the action to update
            status: The new status
            result: The result of the action (optional)
            error_message: Error message if the action failed (optional)
            explanation: Explanation of the status change (optional)
            model_thoughts: List of model thoughts during execution (optional)
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            # Convert string ID to ObjectId if needed
            if isinstance(action_id, str):
                action_id = ObjectId(action_id)
                
            # Prepare update data
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            # Add optional fields if provided
            if result is not None:
                update_data["result"] = result
            if error_message is not None:
                update_data["error_message"] = error_message
            if explanation is not None:
                update_data["explanation"] = explanation
            if model_thoughts is not None:
                update_data["model_thoughts"] = model_thoughts
                
            # Update the document
            result = await self.db.actions.update_one(
                {"_id": action_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating action status: {str(e)}", exc_info=True)
            return False

# Create a singleton instance
mongodb_service = MongoDBService() 
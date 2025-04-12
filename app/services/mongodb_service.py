from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, Union, Dict, Any
import os
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

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
        action_id: Union[str, ObjectId],
        status: str,
        result: Optional[str] = None,
        error_message: Optional[str] = None,
        explanation: Optional[str] = None,
        classification_details: Optional[Dict[str, Any]] = None,
        tts_audio_base64: Optional[str] = None,
    ):
        """Update the status of an action."""
        # Convert string ID to ObjectId if needed
        if isinstance(action_id, str):
            try:
                action_id = ObjectId(action_id)
            except Exception:
                return None
                
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        if result is not None:
            update_data["result"] = result
        if error_message is not None:
            update_data["error_message"] = error_message
        if explanation is not None:
            update_data["explanation"] = explanation
        if classification_details is not None:
            update_data["classification_details"] = classification_details
        if tts_audio_base64 is not None:
            update_data["tts_audio_base64"] = tts_audio_base64

        await self.actions.update_one(
            {"_id": action_id},
            {"$set": update_data}
        )

# Create a singleton instance
mongodb_service = MongoDBService() 
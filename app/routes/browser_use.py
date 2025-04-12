from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent
# from some_agent_library import Agent
# from some_llm_library import ChatOpenAI
# from app.services.langgraph_service import LangGraphService

router = APIRouter(prefix="/api/v1", tags=["browser-use"])

# langgraph_service = LangGraphService()

class BrowserUseRequest(BaseModel):
    prompt: str

class BrowserUseResponse(BaseModel):
    response: str

@router.post("/browser-use", response_model=BrowserUseResponse)
async def handle_browser_use(request: BrowserUseRequest):
    """
    Handle the requests to automatically perform web information retrieval through browser use.
    
    Args:
        request: The user request prompt
        
    Returns:
        BrowserUseResponse: The response from the browser use agent
    """
    try:
        
        agent = Agent(
            task=request.prompt,
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1),
        )

        result = await agent.run()

        agent_message = None
        if result.is_done():
            agent_message = result.history[-1].result[0].extracted_content

        if agent_message is None:
            agent_message = 'There was no response from the browser-use.'

        return BrowserUseResponse(response=agent_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

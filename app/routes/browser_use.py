import os
from fastapi import APIRouter, HTTPException
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from browser_use import Agent

router = APIRouter(prefix="/api/v1", tags=["browser-use"])

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
        # llm_model = 'deepseek-v3'
        # llm_model = 'anthropic'
        llm_model = 'gpt-3.5-turbo'
        # llm_model = 'gpt-4'

        if llm_model == 'deepseek-v3':
            llm=ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-chat', api_key=SecretStr(os.getenv('DEEPSEEK_API_KEY', '')),
                verbose=True)
        elif llm_model == 'anthropic':
            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                temperature=0.0,
                timeout=100, # Increase for complex tasks
                verbose=True,
            )
        elif llm_model == 'gpt-3.5-turbo':
            llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.1, verbose=True)
        elif llm_model == 'gpt-4':
            llm = ChatOpenAI(
                model="gpt-4", 
                temperature=0.1, verbose=True)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        agent = Agent(
            task=request.prompt,
            llm=llm,
            use_vision=False,
            # system_instructions=system_instructions,
        )

        result = await agent.run()

        agent_message = None
        print(result)

        if result.is_done():
            agent_message = result.history[-1].result[0].extracted_content

        if agent_message is None:
            agent_message = 'There was no response from the browser-use.'

        return BrowserUseResponse(response=agent_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

import os
from fastapi import APIRouter, HTTPException
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, SystemPrompt
from typing import Optional

router = APIRouter(prefix="/api/v1", tags=["browser-use"])

# Initialize browser instance
browser = None

class BrowserUseRequest(BaseModel):
    prompt: str

class BrowserUseResponse(BaseModel):
    response: str

class NISystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        # Get existing rules from parent class
        existing_rules = super().important_rules()
        new_rules = ""
        return f'{existing_rules}\n{new_rules}'

async def get_browser() -> Browser:
    """Get or create the browser instance."""
    global browser
    if browser is None:
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path='/usr/bin/brave-browser',
            )
        )
        await browser._init()
        browser.playwright_browser = await browser.get_playwright_browser()
    return browser

async def get_llm(model: str = 'gpt-4') -> tuple[Optional[ChatOpenAI], Optional[ChatOpenAI]]:
    """Get LLM instances based on the specified model."""
    llm = None
    planner_llm = None

    if model == 'deepseek-v3':
        llm = ChatOpenAI(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            api_key=SecretStr(os.getenv('DEEPSEEK_API_KEY', '')),
            verbose=True
        )
    elif model == 'anthropic':
        llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-20240620",
            temperature=0.0,
            timeout=100,
            verbose=True,
        )
    elif model == 'gpt-3.5-turbo':
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            verbose=True
        )
        planner_llm = ChatOpenAI(model='o3-mini')
    elif model == 'gpt-4':
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            verbose=True
        )
        planner_llm = ChatOpenAI(model='o3-mini')
    
    return llm, planner_llm

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
        # Get the existing browser instance or create a new one
        browser = await get_browser()
        
        # llm_model = 'deepseek-v3'
        # llm_model = 'anthropic'
        llm_model = 'gpt-3.5-turbo'
        # llm_model = 'gpt-4'

        # Get LLM instances
        llm, planner_llm = await get_llm(llm_model)
        if not llm:
            raise HTTPException(status_code=400, detail="Failed to initialize LLM")

        initial_actions = [
            {'open_tab': {'url': 'https://www.google.com'}},
        ]
        
        agent = Agent(
            task=request.prompt,
            browser=browser,
            initial_actions=initial_actions,
            llm=llm,
            planner_llm=planner_llm,
            use_vision=False,
            use_vision_for_planner=False,
            system_prompt_class=NISystemPrompt,
            # headless=False,
        )

        result = await agent.run()
        print(f"Browser use result: {result}")

        agent_message = None
        if result.is_done():
            agent_message = result.history[-1].result[0].extracted_content

        if agent_message is None:
            agent_message = 'There was no response from the browser-use.'
        
        return BrowserUseResponse(response=agent_message)
    except Exception as e:
        print(f"Error in browser use: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 

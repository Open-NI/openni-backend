import os
from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig, SystemPrompt
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import logging
import asyncio
from typing import Dict, Any, Optional
import platform
from app.core.config import settings
from browser_use.browser.views import BrowserState
from browser_use.agent.views import AgentOutput

from pydantic import SecretStr

from app.services.mongodb_service import mongodb_service

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This ensures logs go to the console
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Explicitly set the logger level to DEBUG

class CustomBrowserPrompt(SystemPrompt):
    """Custom system prompt for browser automation with specific instructions."""
    
    def important_rules(self) -> str:
        # Get existing rules from parent class
        existing_rules = super().important_rules()
        
        # Add focused rules for browser interaction
        custom_rules = """
9. BROWSER INTERACTION:
   - ALWAYS use the top search bar for searching when appropriate
   - Type queries directly into the search bar and press Enter
   - For non-search tasks, navigate directly to relevant websites
   - Minimize unnecessary clicks and interactions

10. MODAL HANDLING:
    - When encountering cookie consent or modal popups:
      - Look for "Accept", "Allow", "Got it", or similar buttons
      - Click these buttons to dismiss the modal
      - If no clear button is found, try clicking outside the modal
      - If modal persists, try pressing Escape key
      - Continue with the task after dismissing modals

11. RESULT FORMATTING:
    - ALWAYS structure your response in a clear, organized format
    - For search results:
      - Include a summary of findings at the top
      - List results with clear headings and bullet points
      - Include the source URLs and timestamps
      - Highlight key information in bold or quotes
    - For data extraction:
      - Present data in a structured format (tables, lists, etc.)
      - Include source URLs and timestamps
      - Clearly label all extracted information
    - For form submissions:
      - Confirm successful submission
      - Include any confirmation numbers or reference IDs
      - Note any follow-up actions required
    - ALWAYS end with a clear conclusion or summary
    - If no specific data is found, explicitly state that
    - Never return raw HTML or unformatted text

12. If there is data requested with keywords "find", "list", "get", "tell" or similar, you should use the browser to find the data and return it in a structured format always.

"""
        
        return f"{existing_rules}\n{custom_rules}"

class BrowserService:
    """Service for handling browser automation with browser-use."""
    
    def __init__(self):
        """Initialize the browser service."""
        self.agent = None
        self.browser = None
        self.retry_delay = 2  # Delay in seconds between retries
        self.llm, self.planner_llm = self.get_llm(settings, model=settings.MODEL_NAME)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the browser and agent if not already initialized."""
        if self.is_initialized:
            logger.debug("Browser service already initialized")
            return
            
        try:
            browser_path = os.getenv("BROWSER_PATH", "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser")

            if not browser_path or len(browser_path) == 0:
                # Configure browser based on operating system
                system = platform.system()
                if system == "Darwin":  # macOS
                    browser_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
                elif system == "Windows":
                    browser_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
                else:  # Linux and others
                    browser_path = "/usr/bin/brave-browser"

            # Create browser configuration
            self.browser = Browser(
                config=BrowserConfig(
                    chrome_instance_path=browser_path
                )
            )
            
            logger.debug("Browser initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing browser: {str(e)}", exc_info=True)
            raise
    
    async def run_browser(self, task: str, action_id: str = None) -> Dict[str, Any]:
        """
        Run a browser task with the given description using browser-use
        
        Args:
            task: The task description to execute
            
        Returns:
            Dict containing the result from browser-use history
        """
        try:
            # Initialize browser if not already done
            if not self.is_initialized:
                await self.initialize()
            
            #browser_context = BrowserContextConfig(
                #viewport_expansion=1000,
                trace_path="./trace.json"
            #)

            async def update_progress_callback(
                state: BrowserState, model_output: AgentOutput, steps
            ):
                try:
                    if action_id:
                        print(f"Updating screenshot and explanation for action ID: {action_id}. Next goal: {model_output.current_state.next_goal}")
                        screenshot = state.screenshot
                        explanation = model_output.current_state.next_goal

                        # await mongodb_service.update_action_status(action_id, status="browser_task_started", screenshot=screenshot, explanation=explanation)
                except Exception as e:
                    print(f"Error updating action state: {str(e)}", exc_info=True)

            # Create the agent with the task and retry delay
            self.agent = Agent(
                task=task,
                llm=self.llm,
                planner_llm=self.planner_llm,
                retry_delay=self.retry_delay,  # Add delay between retries
                browser=self.browser,  # Use the configured Brave browser
                system_prompt_class=CustomBrowserPrompt,  # Use our custom system prompt
                use_vision=False,
                #register_new_step_callback=update_progress_callback,  # Register the callback for each step
                #browser_context=browser_context
            )
            
            # Run the agent and get results
            history = await self.agent.run()
            
            # Extract the result from the history
            result = None
            
            # Check if the agent is done and extract the result
            if hasattr(history, 'is_done') and history.is_done():
                logger.debug("Agent execution completed successfully")
                
                # Get the last history entry and extract content
                if hasattr(history, 'history') and history.history:
                    last_entry = history.history[-1]
                    logger.debug(f"Last history entry: {last_entry}")
                    
                    if hasattr(last_entry, 'result') and last_entry.result:
                        if isinstance(last_entry.result, list) and last_entry.result:
                            first_result = last_entry.result[0]
                            if hasattr(first_result, 'extracted_content'):
                                result = first_result.extracted_content
                                logger.debug(f"Extracted content from last result")
            
            # If no result found in the last history entry, try other methods
            if not result:
                # Try to get extracted content directly
                if hasattr(history, 'extracted_content') and history.extracted_content:
                    result = history.extracted_content
                    logger.debug(f"Extracted content directly")
                
                # If still no result, try final_result
                if not result and hasattr(history, 'final_result'):
                    result = history.final_result()
                    logger.debug(f"Using final result")
            
            # If we still don't have a result, create a simple one
            if not result:
                result = f"Task completed but no specific result was returned: {task}"
                logger.debug("Created fallback result")
            
            await self.close()
            # Return the result
            return {
                "task": task,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in browser task: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Clean up any resources"""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
                self.is_initialized = False
                logger.debug("Browser closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing browser service: {str(e)}", exc_info=True) 

    def get_llm(self, settings, model: str = 'gpt-4o') -> tuple[Optional[ChatOpenAI], Optional[ChatAnthropic]]:
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
            print("Using DeepSeek v3")
        elif model == 'anthropic':
            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                temperature=0.0,
                timeout=100,
                verbose=True,
            )
            print("Using Anthropic Claude 3")
        elif model == 'gpt-3.5-turbo':
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                verbose=True
            )
            planner_llm = ChatOpenAI(model='o3-mini')
            print("Using GPT-3.5 Turbo, with O3 Mini planner")
        elif model == 'gpt-4o':
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                verbose=True
            )
            planner_llm = ChatOpenAI(model='o3-mini')
            print("Using GPT-4o, with O3 Mini planner")
        else:
            # We try to use openai
            llm = ChatOpenAI(
                model=model,
                temperature=0.1,
                verbose=True
            )
            planner_llm = ChatOpenAI(model='o3-mini')
            print(f"Using OpenAI model: {model}")
        
        return llm, planner_llm

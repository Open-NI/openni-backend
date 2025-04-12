from browser_use import Agent, Browser, BrowserConfig, SystemPrompt
from langchain_openai import ChatOpenAI
import logging
import asyncio
from typing import Dict, Any, Optional
import platform

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
   - Use keyboard shortcuts when possible (Ctrl+L for address bar, Enter to submit)
   - Minimize unnecessary clicks and interactions

10. MODAL HANDLING:
    - When encountering cookie consent or modal popups:
      - Look for "Accept", "Allow", "Got it", or similar buttons
      - Click these buttons to dismiss the modal
      - If no clear button is found, try clicking outside the modal
      - If modal persists, try pressing Escape key
      - Continue with the task after dismissing modals

11. TASK COMPLETION:
    - Always verify that the task has been completed successfully
    - Extract and return relevant information as requested
    - For search tasks, return top results with titles, URLs, and snippets
    - For form submissions, confirm the submission was successful
    - For data extraction, ensure all required data is captured
"""
        
        return f"{existing_rules}\n{custom_rules}"

class BrowserService:
    """Service for handling browser automation with browser-use."""
    
    def __init__(self):
        """Initialize the browser service."""
        self.agent = None
        self.browser = None
        self.retry_delay = 2  # Delay in seconds between retries
        self.llm = ChatOpenAI(model='gpt-4o')
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the browser and agent if not already initialized."""
        if self.is_initialized:
            logger.debug("Browser service already initialized")
            return
            
        try:
            # Configure browser based on operating system
            system = platform.system()
            if system == "Darwin":  # macOS
                brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
            elif system == "Windows":
                brave_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
            else:  # Linux and others
                brave_path = "/usr/bin/brave-browser"
                
            # Create browser configuration
            self.browser = Browser(
                config=BrowserConfig(
                    chrome_instance_path=brave_path
                )
            )
            
            logger.debug("Browser initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing browser: {str(e)}", exc_info=True)
            raise
    
    async def run_browser(self, task: str) -> Dict[str, Any]:
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
            
            # Create the agent with the task and retry delay
            self.agent = Agent(
                task=task,
                llm=self.llm,
                retry_delay=self.retry_delay,  # Add delay between retries
                browser=self.browser,  # Use the configured Brave browser
                system_prompt_class=CustomBrowserPrompt,  # Use our custom system prompt
                use_vision=False
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
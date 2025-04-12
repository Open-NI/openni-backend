from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import logging
import asyncio
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BrowserService:
    """Service for handling browser automation with browser-use."""
    
    def __init__(self):
        """Initialize the browser service."""
        self.agent = None
        self.retry_delay = 2  # Delay in seconds between retries
    
    async def run_browser(self, query: str) -> Dict[str, Any]:
        """
        Run a browser search with the given query using browser-use
        
        Args:
            query: The search query to execute
            
        Returns:
            Dict containing the result from browser-use history
        """
        try:
            # Create the agent with the search task and retry delay
            self.agent = Agent(
                task=f"Search for '{query}' on Google and return the top 5 results with their titles, URLs, and snippets.",
                llm=ChatOpenAI(model='gpt-4o'),
                retry_delay=self.retry_delay  # Add delay between retries
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
                result = f"No results found for query: {query}"
                logger.debug("Created fallback result")
            
            # Return the result
            return {
                "query": query,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in browser search: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Clean up any resources"""
        try:
            if self.agent:
                # Clean up agent resources if needed
                self.agent = None
            logger.debug("Browser service resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing browser service: {str(e)}", exc_info=True) 
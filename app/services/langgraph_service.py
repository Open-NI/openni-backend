from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, List, Tuple, Optional, Any, TypedDict, Annotated, Literal
from app.core.config import settings
from app.models.classification import ClassificationLabel
from fastapi import HTTPException
import logging
import json
import re
from app.services.browser_service import BrowserService
from enum import Enum

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

# Define API actions as an enum
class ApiAction(Enum):
    """Available API actions that can be performed by the system."""
    NONE = "none"  # No API action needed
    GET_CURRENT_DATETIME = "GET_CURRENT_DATETIME"  # Get current date and time

# Define API action handler functions
def get_current_datetime(params=None):
    """Get the current date and time."""
    from datetime import datetime
    current_datetime = datetime.now()
    return f"The current date and time is: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}"

# Dictionary mapping API actions to their handler functions
API_ACTION_HANDLERS = {
    "GET_CURRENT_DATETIME": get_current_datetime
}

# Define the state schema
class GraphState(TypedDict):
    """State schema for the classification graph."""
    text: str
    classification: Optional[str]
    response: Optional[str]
    browser_input: Optional[str]
    browser_result: Optional[str]
    api_action: Optional[str]
    api_params: Optional[Dict[str, Any]]

class LangGraphService:
    """Service for handling text classification using LangGraph."""
    
    def __init__(self):
        """Initialize the LangGraph service."""
        self.llm = ChatOpenAI(model='gpt-4o')
        self.browser_service = BrowserService()
        
        # Classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a classifier that determines how to handle user requests.
            
            You have three possible classifications:
            1. "api_action" - The request requires using one of our API actions
            2. "normal_response" - The request can be answered directly without external tools
            3. "browser_use" - The request requires searching the web for information
            
            Available API actions:
            {api_actions}

            If the request matches one of the API actions, classify it as "api_action" and specify which action to use.
            If the request can be answered directly, classify it as "normal_response".
            If the request requires searching the web, classify it as "browser_use".
            
            Return your classification as a JSON object with the following structure:
            {{
                "classification": "api_action" | "normal_response" | "browser_use",
                "api_action": "action_name" | null,
                "api_params": {{param1: value1, ...}} | null,
                "browser_input": "search query" | null
            }}
            
            IMPORTANT: Return ONLY the JSON object without any markdown formatting, code blocks, or additional text.
            """),
            ("human", "{text}")
        ])
        
        # Response generation prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Provide a clear, informative, and helpful response to the user's query.
            Be concise but thorough. If you don't know something, say so rather than making up information.
            Your responses MUST be extremely brief and concise. Use 30 completion_tokens or less. Focus only on the most essential information. No explanations, no context, no elaboration. Just the direct answer.
            """),
            ("human", "{text}")
        ])
        
        # Browser query enhancement prompt
        self.browser_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a search query optimizer. Your task is to enhance the user's input to make it more effective for web searching.
            - Add relevant context and specifics
            - Include important keywords
            - Make the query more explicit and search-friendly
            - Keep the core meaning of the original query
            - Don't add any explanations or additional text, just return the enhanced query
            
            If you encounter a login popup or paywall during the search, end the search immediately and return a message indicating that the content requires authentication.
            
            Example:
            Input: "weather in London"
            Output: "current weather forecast London UK temperature precipitation"
            
            Input: "latest iPhone price"
            Output: "iPhone 15 Pro Max current retail price US market 2024"
            """),
            ("human", "{text}")
        ])
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the classification graph."""
        # Define the classification node
        def classify_node(state: GraphState) -> GraphState:
            try:
                # Get the input text
                text = state.get("text", "")
                logger.debug(f"Classifying text: {text}")
                
                # Get classification from LLM
                messages = self.classification_prompt.format_messages(
                    text=text,
                    api_actions="\n".join([f"- {action.name}: {action.value}" for action in ApiAction])
                )
                response = self.llm.invoke(messages)
                
                # Parse the JSON response
                try:
                    # Clean the response content to remove any markdown formatting
                    content = response.content.strip()
                    # Remove markdown code block if present
                    if content.startswith("```") and content.endswith("```"):
                        content = content.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                        # Remove language identifier if present (e.g., ```json)
                        if content.startswith("json"):
                            content = content[4:].strip()
                    
                    logger.debug(f"Cleaned response content: {content}")
                    result = json.loads(content)
                    logger.debug(f"Classification result: {result}")
                    
                    # Update state with classification and any API action details
                    state["classification"] = result.get("classification", "normal_response")
                    
                    # Ensure classification matches the expected literal types
                    if state["classification"] == "api_action":
                        state["classification"] = "api_actions"  # Convert to the correct enum value
                    
                    state["api_action"] = result.get("api_action")
                    state["api_params"] = result.get("api_params")
                    state["browser_input"] = result.get("browser_input")
                    
                    # If this is an API action, execute it directly here
                    if state["classification"] == "api_actions" and state["api_action"]:
                        api_action = state["api_action"]
                        api_params = state["api_params"] or {}
                        
                        # Log the API action and parameters
                        logger.debug(f"Executing API action: {api_action} with parameters: {api_params}")
                        
                        # Check if the API action exists in our handlers
                        if api_action in API_ACTION_HANDLERS:
                            # Execute the appropriate API action using the handler function
                            handler_function = API_ACTION_HANDLERS[api_action]
                            state["response"] = handler_function(api_params)
                            logger.debug(f"API action executed successfully: {state['response']}")
                        else:
                            # Handle unknown API action
                            logger.warning(f"Unknown API action: {api_action}")
                            state["response"] = f"I'm sorry, I don't know how to handle the API action: {api_action}"
                            # Reset classification to normal_response since we couldn't handle the API action
                            state["classification"] = "normal_response"
                    
                    return state
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response.content}")
                    state["classification"] = "normal_response"
                    return state
                    
            except Exception as e:
                logger.error(f"Error in classification node: {str(e)}", exc_info=True)
                state["classification"] = "normal_response"
                return state
        
        # Define the response generation node
        def generate_response_node(state: GraphState) -> GraphState:
            try:
                # Get the input text
                text = state.get("text", "")
                logger.debug(f"Generating response for: {text}")
                
                # Generate response from LLM
                messages = self.response_prompt.format_messages(text=text)
                response = self.llm.invoke(messages)
                
                # Update state with response
                state["response"] = response.content.strip()
                logger.debug(f"Generated response: {state['response']}")
                return state
                
            except Exception as e:
                logger.error(f"Error in response generation: {str(e)}", exc_info=True)
                state["response"] = "I apologize, but I encountered an error processing your request."
                return state
        
        # Define the browser query enhancement node
        def browser_query_node(state: GraphState) -> GraphState:
            try:
                # Get the input text
                text = state.get("text", "")
                logger.debug(f"Enhancing browser query for: {text}")
                
                # Generate enhanced browser query from LLM
                messages = self.browser_prompt.format_messages(text=text)
                response = self.llm.invoke(messages)
                
                # Update state with enhanced browser query
                state["browser_input"] = response.content.strip()
                logger.debug(f"Enhanced browser query: {state['browser_input']}")
                return state
                
            except Exception as e:
                logger.error(f"Error in browser query enhancement: {str(e)}", exc_info=True)
                state["browser_input"] = text  # Fallback to original text
                return state
        
        # Define the router function
        def router(state: GraphState) -> Literal["generate_response", "browser_query", END]:
            classification = state.get("classification", "normal_response")
            logger.debug(f"Routing based on classification: {classification}")
            
            if classification == "normal_response":
                return "generate_response"
            elif classification == "browser_use":
                # For browser_use, we'll just return the enhanced query without executing it
                logger.debug("Browser use detected, returning enhanced query without execution")
                return END
            elif classification == "api_actions":
                # For API actions, we've already executed the action in the classification node
                # and set the response, so we can end the graph
                logger.debug("API action already executed, ending graph")
                return END
            else:
                logger.warning(f"Unknown classification: {classification}, defaulting to generate_response")
                return "generate_response"
        
        # Build the graph with state schema
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("classify", classify_node)
        workflow.add_node("generate_response", generate_response_node)
        workflow.add_node("browser_query", browser_query_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify",
            router,
            {
                "generate_response": "generate_response",
                "browser_query": "browser_query",
                END: END
            }
        )
        
        # Add edges
        workflow.add_edge("browser_query", END)
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        return workflow.compile()
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process the input text through the LangGraph workflow.
        
        Args:
            text: The text to process
            
        Returns:
            Dict containing the classification, response, and any browser results
        """
        try:
            # Initialize state with input text
            initial_state = {"text": text}
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract results
            classification = final_state.get("classification", "normal_response")
            response = final_state.get("response", "")
            browser_input = final_state.get("browser_input", "")
            api_action = final_state.get("api_action", None)
            api_params = final_state.get("api_params", None)
            
            # Prepare result
            result = {
                "classification": classification,
                "response": response,
                "browser_input": browser_input
            }
            
            # Add API action and parameters if available
            if api_action:
                result["api_action"] = api_action
                result["api_params"] = api_params
            
            # For api_actions, ensure we have a response
            if classification == "api_actions" and not response:
                result["response"] = f"I'm sorry, I couldn't execute the API action: {api_action}"
            
            logger.debug(f"Final results: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}", exc_info=True)
            return {
                "classification": "normal_response",
                "response": "I apologize, but I encountered an error processing your request."
            }
    
    async def _generate_response(self, text: str) -> str:
        """Generate a response for the input text using the language model."""

        response = self.llm.invoke(
            [
                {"role": "system", "content": "You are a helpful assistant. Your responses MUST be extremely brief and concise. Focus only on the most essential information. No explanations, no context, no elaboration. Just the direct answer."},
                {"role": "user", "content": text}
            ],
            max_tokens=20
        )
        return response.content
    
    def _generate_browser_input(self, text: str) -> str:
        """
        Generate an enhanced search query for browser use.
        
        Args:
            text: The original input text
            
        Returns:
            str: The enhanced search query
        """
        system_prompt = """You are a search query optimizer. Your task is to enhance the user's input to make it more effective for web searching.
        - Add relevant context and specifics
        - Include important keywords
        - Make the query more explicit and search-friendly
        - Keep the core meaning of the original query
        - Don't add any explanations or additional text, just return the enhanced query
        
        Example:
        Input: "weather in London"
        Output: "current weather forecast London UK temperature precipitation"
        
        Input: "latest iPhone price"
        Output: "iPhone 15 Pro Max current retail price US market 2024"
        """
        
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ])
        
        return response.content.strip()
    
    async def close(self):
        """Clean up resources."""
        await self.browser_service.close() 
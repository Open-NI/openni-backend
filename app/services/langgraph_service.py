from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, List, Tuple, Optional, Any, TypedDict, Annotated, Literal
from app.core.config import settings
from app.models.classification import ClassificationLabel
from app.models.assistant import AssistantVoice
from fastapi import HTTPException
import logging
import json
from app.services.browser_service import BrowserService
from enum import Enum
import requests

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

# Global variable to store conversation history
CONVERSATION_HISTORY = []

# Define API actions as an enum
class ApiAction(Enum):
    """Available API actions that can be performed by the system."""
    NONE = "none"  # No API action needed
    GET_CURRENT_DATETIME = "GET_CURRENT_DATETIME"  # Get current date and time
    MEME_CAPTION = "MEME_CAPTION"  # Generate meme captions

# Define API action handler functions
def get_current_datetime(params=None):
    """Get the current date and time."""
    from datetime import datetime
    current_datetime = datetime.now()
    return f"The current date and time is {current_datetime.strftime('%A, %B %-d, %Y at %-I:%M %p')}."

def generate_meme_caption(params=None):
    """Generate meme captions based on user request."""
    if not params or "text" not in params:
        return "Please provide text for the meme caption."
    
    logger.debug(f"Generating meme caption for: {params['text']}")

    # Available meme templates
    meme_templates = {
        "217743513": "Uno draw cards",
        "181913649": "Drake hotline bling",
        "129242436": "Change My Mind",
        "188390779": "Woman Yelling At Cat",
        "224015000": "Bernie Sanders Once Again Asking",
        "77045868": "Pawn Stars Best I Can Do"
    }
    
    # Use LLM to determine the best meme template and generate captions
    prompt = f"""Based on the following text, select the most appropriate meme template and generate captions for it.
    Text: {params['text']}
    
    Available templates:
    {json.dumps(meme_templates, indent=2)}
    
    Return a JSON object with:
    - template_id: The ID of the chosen template
    - top_text: The text for the top of the meme
    - bottom_text: The text for the bottom of the meme
    
    Make the captions funny and relevant to the input text."""
    
    response = ChatOpenAI(model='gpt-3.5-turbo').invoke(prompt)
    
    try:
        result = json.loads(response.content)
        
        # Call the Imgflip API to generate the meme
        api_url = "https://api.imgflip.com/caption_image"
        api_params = {
            "username": "mk1340",
            "password": "fortnite123",
            "template_id": result["template_id"],
            "text0": result["top_text"],
            "text1": result["bottom_text"]
        }
        
        api_response = requests.post(api_url, data=api_params)
        api_data = api_response.json()
        
        if api_data["success"]:
            return api_data["data"]["url"]
        else:
            return f"Failed to generate meme: {api_data['error_message']}"
            
    except json.JSONDecodeError:
        return "Failed to generate meme captions."
    except Exception as e:
        logger.error(f"Error generating meme: {str(e)}")
        return f"Error generating meme: {str(e)}"

# Dictionary mapping API actions to their handler functions
API_ACTION_HANDLERS = {
    "GET_CURRENT_DATETIME": get_current_datetime,
    "MEME_CAPTION": generate_meme_caption
}

# Define the state schema
class GraphState(TypedDict):
    """State schema for the classification graph."""
    text: str
    voice: Optional[str]
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

        self.history_state = []
        
        # Classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a classifier that determines how to handle user requests.
            
            You have three possible classifications:
            1. "api_action" - The request requires using one of our API actions
            2. "normal_response" - The request can be answered directly without external tools
            3. "browser_use" - The request requires searching the web for information

            Under no circustances whatsoever should you return anything else besides one of these 3 options.
            
            Available API actions:
            {api_actions}

            If the request matches one of the API actions, classify it as "api_action" and specify which action to use.
            If the request can be answered directly, classify it as "normal_response".
            If the request requires searching and or requesting web actions or opening a browser, classify it as "browser_use" and return the relevant "search query" to be used for the browser, effective for web searching.
            - Make the query more explicit and search-friendly
            - Keep the core meaning of the original query
            - DO NOT add any dates or years to the "search query", instead use relative terms like "current", "latest", "next week", "last month", etc.
            If there are explicit keywords like "find", "get", "write" or similar, you should include those in the search query.
            If the request requires data extraction from a website, classify it as "data_extraction" and return the relevant "search query" to be used for the browser, effective for web searching.
            Return your classification as a JSON object with the following structure:
            {{
                "classification": "api_action" | "normal_response" | "browser_use",
                "api_action": "action_name" | null,
                "api_params": {{param1: value1, ...}} | null,
                "browser_input": "search query" | null
            }}

Example: User: "I feel depressed. I want to go for a trip somewhere away for vacation".
              (User is implying you should book a flight for them, but did not specify which airline, date or location, so before  performing an action, ask him about specifics.)
         Response:

             {{
                "classification": "normal_response" ,
                "api_action": null,
                "api_params": null,
                "browser_input": null
            }} 

        User: "Somewhere sunny, far away from Slovenia."

        Response: 

             {{
                "classification": "browser_use" ,
                "api_action": null,
                "api_params": null,
                "browser_input": "Book a flight next week from Ljubljana to Bangkok."
            }} 


Example: User: "I dont feel so good, I'm unemployed."
              (User is implying he is unemployed so you should help him find a job, but you dont know his skillset or prior work experience.)
           Response:   {{
                "classification": "normal_response" ,
                "api_action": null,
                "api_params": null,
                "browser_input": null
            }}
            User: "Im a spring developer"

            Response   {{
                "classification": "browser_use" ,
                "api_action": null,
                "api_params": null,
                "browser_input": "Search Linkedin for Spring developer jobs and apply to them. Write messages to people who are hiring."
            }} 
            
IMPORTANT: Return ONLY the JSON object without any markdown formatting, code blocks, or additional text.
            """),
            ("human", "{text}")
        ])
        
        # Response generation prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Provide a clear, informative, and helpful response to the user's query.
            Be concise but thorough. If you don't know something, say so rather than making up information.
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
            - DO NOT add any explanations or additional text, dates or years, instead use relative terms like "current", "latest", "next week", "last month", etc.
            
            If you encounter a login popup or paywall during the search, end the search immediately and return a message indicating that the content requires authentication.
            
            Example:
            Input: "weather in London"
            Output: "current weather forecast London UK temperature precipitation"
            
            Input: "latest iPhone price"
            Output: "iPhone 15 Pro Max current retail price US market"
            
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
                print(messages)
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

                        if api_action == "MEME_CAPTION":
                            api_params = { "text": text }
                        else:
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
        async def generate_response_node(state: GraphState) -> GraphState:
            """Generate a response for the input text."""
            try:
                # Get the input text and voice
                text = state.get("text", "")
                voice = state.get("voice", "af_heart")
                
                # Generate response using the language model with voice personalization
                response = await self._generate_response(text, voice)

                # Update state with response
                state["response"] = response
                return state
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                state["response"] = "I apologize, but I encountered an error generating a response."
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
    
    async def process_text(self, text: str, voice: str = "af_heart") -> Dict[str, Any]:
        """
        Process the input text through the LangGraph workflow.
        
        Args:
            text: The text to process
            voice: The voice to use for personalization (default: "af_heart")
            
        Returns:
            Dict containing the classification, response, and any browser results
        """
        try:
            # Initialize state with input text
            initial_state = {"text": text, "voice": voice}
            
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
    
    async def _generate_response(self, text: str, voice: str = "af_heart") -> str:
        """
        Generate a response for the input text using the language model.
        
        Args:
            text: The input text to generate a response for
            voice: The voice to use for personalization (default: "af_heart")
            
        Returns:
            str: The generated response
        """
        # Get the personality trait based on the voice
        personality_trait = AssistantVoice.get_personality_trait(voice)
        
        response = self.llm.invoke(
            [
                {"role": "system", "content": personality_trait},
                *CONVERSATION_HISTORY,
                {"role": "user", "content": text}
            ]
        )

        CONVERSATION_HISTORY.append({"role": "user", "content": text})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": response.content})
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
        - Don't add any explanations or additional text, dates, years, just return the enhanced query
        
        Example:
        Input: "weather in London"
        Output: "current weather forecast London UK temperature precipitation"
        
        Input: "latest iPhone price"
        Output: "iPhone 15 Pro Max current retail price US market"
        """
        
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ])
        
        return response.content.strip()
    
    async def close(self):
        """Clean up resources."""
        await self.browser_service.close() 
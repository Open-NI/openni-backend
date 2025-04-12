from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, List, Tuple, Optional, Any
from app.core.config import settings
from app.models.classification import ClassificationLabel
from fastapi import HTTPException
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LangGraphService:
    """Service for handling LangGraph and OpenAI operations."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Define the classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a classification agent that determines how to respond to user queries.
            
            Classification rules:
            - Use 'browser_use' if:
              * The query requires looking up current information
              * The query asks for facts you're unsure about
              * The query explicitly requests searching for information
              * The query contains keywords like "find", "search", "look up", "what is", "who is", "when", "where"
            
            - Use 'api_action' if:
              * The query requires interacting with an external API
              * The query involves data processing or transformation
              * The query contains keywords like "calculate", "process", "analyze", "convert"
            
            - Use 'normal_response' for:
              * General conversation
              * Questions you can answer without external information
              * Clarifications or follow-ups
            
            Return your response in the following JSON format:
            {
                "classification": "browser_use|api_action|normal_response",
                "response": "Your response here",
                "browser_input": "Search query if classification is browser_use, null otherwise"
            }
            
            Be concise and direct in your responses."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        # Define the nodes
        def classify_node(state):
            try:
                logger.debug("Running classification node")
                messages = self.classification_prompt.format_messages(
                    input=state["input"],
                    chat_history=state.get("chat_history", [])
                )
                response = self.llm.invoke(messages)
                result = json.loads(response.content)
                logger.debug(f"Classification result: {result}")
                return result
            except Exception as e:
                logger.error(f"Error in classification node: {str(e)}", exc_info=True)
                raise
        
        def response_node(state):
            try:
                logger.debug("Running response node")
                return {"response": state["response"]}
            except Exception as e:
                logger.error(f"Error in response node: {str(e)}", exc_info=True)
                raise
        
        # Build the graph
        workflow = StateGraph(StateType=Dict)
        
        # Add nodes
        workflow.add_node("classify", classify_node)
        workflow.add_node("response", response_node)
        
        # Add edges
        workflow.add_conditional_edges(
            "classify",
            lambda x: "response",
            {
                "response": "response"
            }
        )
        workflow.add_edge("response", END)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        return workflow.compile()
    
    async def classify_text(self, text: str) -> Tuple[str, str, Optional[str]]:
        """
        Classify the input text and return a tuple of (classification, response, browser_input).
        
        Args:
            text: The input text to classify
            
        Returns:
            Tuple containing:
            - classification: The classification of the input
            - response: A response to the input
            - browser_input: The search query if classification is 'browser_use', None otherwise
        """
        try:
            logger.debug(f"Classifying text: {text}")
            result = self.graph.invoke({"input": text})
            logger.debug(f"Graph result: {result}")
            
            if not result:
                logger.error("No result returned from graph")
                return "normal_response", "I couldn't process your request.", None
            
            classification = result.get("classification", "normal_response")
            response = result.get("response", "I couldn't generate a response.")
            browser_input = result.get("browser_input")
            
            return classification, response, browser_input
            
        except Exception as e:
            logger.error(f"Error in classify_text: {str(e)}", exc_info=True)
            return "normal_response", f"An error occurred: {str(e)}", None
    
    def _generate_response(self, text: str) -> str:
        """
        Generate a response for the input text.
        
        Args:
            text: The input text
            
        Returns:
            str: The generated response
        """
        response = self.llm.invoke([
            {"role": "system", "content": "You are a helpful assistant. Provide a clear and concise response to the user's request."},
            {"role": "user", "content": text}
        ])
        
        return response.content.strip()
    
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
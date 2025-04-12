from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import os
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the classification labels
ClassificationLabel = Literal["browser_use", "normal_response", "api_actions"]

# Define request and response models
class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    classification: ClassificationLabel
    explanation: str

# Initialize FastAPI app
app = FastAPI()

# Define the classification function
def classify_text(text: str) -> ClassificationLabel:
    system_prompt = """You are a classifier that determines how to handle user requests. 
    Classify the request into one of these categories:
    - "browser_use": If the request needs to extract information from the web
    - "normal_response": If the request can be answered using general knowledge
    - "api_actions": If the request requires specific API actions
    
    Respond with ONLY the classification label, nothing else."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.1
    )
    
    classification = response.choices[0].message.content.strip().lower()
    
    # Validate the classification
    if classification not in ["browser_use", "normal_response", "api_actions"]:
        raise HTTPException(status_code=500, detail="Invalid classification from LLM")
    
    return classification

@app.post("/classify", response_model=ClassificationResponse)
async def classify_request(request: ClassificationRequest):
    try:
        classification = classify_text(request.text)
        
        # Generate explanation for the classification
        explanation_prompt = f"""Explain why the text "{request.text}" was classified as "{classification}".
        Keep the explanation concise and clear."""
        
        explanation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": explanation_prompt}
            ],
            temperature=0.7
        )
        
        explanation = explanation_response.choices[0].message.content.strip()
        
        return ClassificationResponse(
            classification=classification,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
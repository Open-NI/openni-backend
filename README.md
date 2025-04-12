# LLM Text Classifier API

This FastAPI application provides an endpoint for classifying text using OpenAI's GPT model. The classifier determines whether a given text should be handled as:
- Browser use (for web information extraction)
- Normal response (for general knowledge responses)
- API actions (for specific API operations)

## Project Structure

```
app/
├── core/
│   ├── __init__.py
│   └── config.py         # Application configuration
├── models/
│   ├── __init__.py
│   └── classification.py # Pydantic models
├── routes/
│   ├── __init__.py
│   └── classification.py # API routes
├── services/
│   ├── __init__.py
│   └── langgraph_service.py # LangGraph and OpenAI service
├── __init__.py
└── main.py              # FastAPI application
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

Start the server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /api/v1/classify

Classifies the input text and provides an explanation for the classification.

Request body:
```json
{
    "text": "Your text to classify here"
}
```

Response:
```json
{
    "classification": "browser_use|normal_response|api_actions",
    "explanation": "Explanation of why the text was classified this way"
}
```

## Example Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={"text": "What's the weather like in London?"}
)
print(response.json())
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 
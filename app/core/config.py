import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    APP_NAME: str = "LLM Text Classifier API"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    
    class Config:
        env_file = ".env"

    def get_llm(self):
        if self.MODEL_NAME == "gpt-3.5-turbo":
            return ChatOpenAI(model="gpt-3.5-turbo", api_key=self.OPENAI_API_KEY)
        elif self.MODEL_NAME == "gpt-4o":
            return ChatOpenAI(model="gpt-4o", api_key=self.OPENAI_API_KEY)
        

settings = Settings()

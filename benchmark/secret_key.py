"""
API keys loaded from environment variables.
Set these in the project root .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load from project root .env
load_dotenv(Path(__file__).parent.parent / ".env")

HF_KEY = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
import os

# importing dot env
from dotenv import load_dotenv

# loading the environment variables
load_dotenv()

MONGO_CONNECTION_URL = os.getenv("MONGO_CONNECTION_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# fine-tuned model served locally by Ollama (see docker-compose.yml)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "anindaghosh/cs-gy-6613-rag-project")

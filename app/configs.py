import os

# importing dot env
from dotenv import load_dotenv

# loading the environment variables
load_dotenv()

MONGO_CONNECTION_URL = os.getenv("MONGO_CONNECTION_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

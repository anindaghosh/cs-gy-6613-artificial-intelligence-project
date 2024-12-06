import re
import pymongo
import requests
import re
import os
from bs4 import BeautifulSoup
import shutil
import subprocess

from loguru import logger

from tempfile import mkdtemp
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from clearml import Task

from configs import MONGO_CONNECTION_URL


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def medium_cleaner():

    client = pymongo.MongoClient(MONGO_CONNECTION_URL)
    db = client["rag"]
    collection = db["rag_raw_data"]

    posts = collection.find({"platform": "medium"}).limit(1)

    for post in posts:
        content = post["content"]
        title = post["title"]
        subtitle = post["subtitle"] if post["subtitle"] else ""

        valid_content = [title, subtitle, content]

        logger.info(f"Cleaning content for post: {post['url']}")
        logger.info(f"Original content: {content}")
        cleaned_content = clean_text(" #### ".join(valid_content))
        logger.info(f"Cleaned content: {cleaned_content}")


if __name__ == "__main__":
    medium_cleaner()

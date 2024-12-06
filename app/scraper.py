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


class MediumScraper:

    def scrape(self, link: str):

        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(MONGO_CONNECTION_URL)
            db = client["rag"]
            collection = db["rag_raw_data"]

            logger.info(f"Starting scraping Medium article: {link}")

            if collection.find_one({"url": link}):
                logger.info(f"Article already exists in the database: {link}")
                return

            # Request the page
            res = requests.get(link)
            res.raise_for_status()

            # Parse the page
            soup = BeautifulSoup(res.text, "html.parser")

            # Extract article details

            title = soup.find_all("h1", class_="pw-post-title")
            subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")
            author = soup.find("meta", {"name": "author"})["content"]

            publication_date = soup.find(
                "meta", {"property": "article:published_time"}
            )["content"]

            # Save the page
            collection.insert_one(
                {
                    "url": link,
                    "html": res.text,
                    "title": title[0].string if title else None,
                    "subtitle": subtitle[0].string if subtitle else None,
                    "author": author,
                    "publication_date": publication_date,
                    "content": soup.get_text(),
                    "platform": "medium",
                }
            )

            # Close the MongoDB connection
            client.close()

        except Exception as e:
            logger.error(f"Error fetching Medium article: {link}")
            logger.error(e)
            return None

        finally:
            logger.info(f"Finished scraping Medium article: {link}")


class GithubScraper:

    def scrape(self, link: str):

        ignore_file_types = (".git", ".toml", ".lock", ".png", ".jpg", ".jpeg", ".svg")

        # Connect to MongoDB

        client = pymongo.MongoClient(MONGO_CONNECTION_URL)
        db = client["rag"]
        collection = db["rag_raw_data"]

        # check if link already exists in database
        if collection.find_one({"url": link}):
            logger.info(f"Repository already exists in the database: {link}")
            return

        # Request the page
        logger.info(f"Starting scraping GitHub repository: {link}")
        repo_name = link.rstrip("/").split("/")[-1]
        local_temp = mkdtemp()

        try:
            os.chdir(local_temp)
            subprocess.run(["git", "clone", link])

            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])  #
            tree = {}
            for root, _, files in os.walk(repo_path):
                dir = root.replace(repo_path, "").lstrip("/")
                if dir.startswith(ignore_file_types):
                    continue
                for file in files:
                    if file.endswith(ignore_file_types):
                        continue
                    file_path = os.path.join(dir, file)
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        tree[file_path] = f.read().replace(" ", "")

            # write to database

            collection.insert_one(
                {
                    "url": link,
                    "name": repo_name,
                    "content": tree,
                    "platform": "github",
                }
            )

        except Exception as e:
            logger.error(f"Error scraping GitHub repository: {link}")
            logger.error(e)
        finally:
            shutil.rmtree(local_temp)
            logger.info(f"Finished scraping GitHub repository: {link}")


class YoutubeScraper:

    client = pymongo.MongoClient(MONGO_CONNECTION_URL)
    db = client["rag"]
    collection = db["rag_raw_data"]

    def extract_transcripts(self, video_id):
        """
        Extract transcript from a YouTube video.
        """
        try:
            # Check if video id exists in the database
            if self.collection.find_one({"video_id": video_id}):
                logger.info(f"Video already exists in the database: {video_id}")
                return

            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return " ".join([entry["text"] for entry in transcript])

        except Exception as e:
            logger.error(f"Error fetching transcript for {video_id}: {e}")
            return None

        finally:
            logger.info(f"Finished scraping Youtube video: {link}")

    def preprocess_transcript(self, transcript):
        """
        Clean and preprocess the transcript text.
        """
        # Remove non-alphanumeric characters
        transcript = re.sub(r"[^A-Za-z0-9\s]", "", transcript)
        return transcript.lower()

    def load_to_mongodb(
        self,
        video_id,
        link,
        transcript,
    ):
        """
        Store transcript in MongoDB.
        """

        try:
            data = {
                "video_id": video_id,
                "content": transcript,
                "platform": "youtube",
                "url": link,
            }
            self.collection.insert_one(data)
            logger.info(f"Transcript for {video_id} stored successfully!")

        except Exception as e:
            logger.error(f"Error storing transcript for {video_id}: {e}")

    def scrape(self, link: str):

        try:
            parsed_url = urlparse(link)
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get("v", [None])[0]

            transcript = self.extract_transcripts(video_id)

            if transcript:
                processed_transcript = self.preprocess_transcript(transcript)

                if processed_transcript:
                    self.load_to_mongodb(video_id, link, processed_transcript)

        except Exception as e:
            logger.error(f"Error scraping YouTube video: {link}")
            logger.error(e)

        finally:
            logger.info(f"Finished scraping Youtube video: {link}")


if __name__ == "__main__":

    task = Task.init(project_name="cs-gy-6613-rag", task_name="scraper")

    logger.info("Starting scraping...")

    medium_scraper = MediumScraper()
    github_scraper = GithubScraper()
    youtube_scraper = YoutubeScraper()

    # find all the links in the database under collection media_urls and scrape them based on platform key value

    client = pymongo.MongoClient(MONGO_CONNECTION_URL)
    db = client["rag"]
    collection = db["media_urls"]

    urls = collection.find({"platform": "medium"})

    for doc in urls:
        link = doc["url"]
        platform = doc["platform"]

        if platform == "medium":
            medium_scraper.scrape(link)
        elif platform == "github":
            github_scraper.scrape(link)
        elif platform == "youtube":
            youtube_scraper.scrape(link)

    logger.info("Finished scraping.")

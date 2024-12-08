import re
import re

from loguru import logger

from clearml import Task

from app.helpers.mongo_client import get_mongo_client


class ETL_Clean:

    mongo_client = get_mongo_client()

    def clean_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def medium_cleaner(self, post, cleaned_collection):

        content = post["content"]
        title = post["title"]
        subtitle = (
            post["subtitle"] if post["subtitle"] else ""
        )  # Handle missing subtitles

        # Combine content pieces for cleaning
        valid_content = [title, subtitle, content]

        logger.info(f"Cleaning content for post: {post['url']}")
        # logger.info(f"Original content: {content}")

        # Clean the combined content
        cleaned_content = self.clean_text(" #### ".join(valid_content))

        # logger.info(f"Cleaned content: {cleaned_content}")

        # Prepare the cleaned post data to be inserted into the cleaned collection
        cleaned_post = {
            "url": post["url"],  # Keep the URL to identify the post
            "cleaned_content": cleaned_content,  # The cleaned content
            # "original_content": content,  # Optionally store the original content
            # "title": title,  # Optionally store the title
            # "subtitle": subtitle,  # Optionally store the subtitle
            "platform": post["platform"],  # Keep the platform name for reference
        }

        # Insert the cleaned post into the 'rag_cleaned_data' collection
        cleaned_collection.insert_one(cleaned_post)

        # Optionally, update the original collection with the cleaned content
        # collection.update_one({"_id": post["_id"]}, {"$set": {"cleaned_content": cleaned_content}})

        logger.info(f"Content cleaning and saving complete for post: {post['url']}")

    def youtube_cleaner(self, post, cleaned_collection):
        """
        Connects to MongoDB Atlas, retrieves posts from the 'rag_raw_data' collection,
        cleans the content, and stores the cleaned data in the 'rag_cleaned_data' collection.
        """

        # Safely get 'video_id', 'content', and 'url' fields using 'get()' to avoid KeyError
        video_id = post.get(
            "video_id", ""
        )  # Default to an empty string if 'video_id' is missing
        content = post.get(
            "content", ""
        )  # Default to an empty string if 'content' is missing
        url = post.get("url", "")  # Default to an empty string if 'url' is missing

        # Log information about the cleaning process
        logger.info(f"Cleaning content for video: {url}")
        # logger.info(f"Original content: {content}")

        # Clean the content
        cleaned_content = self.clean_text(content)

        # logger.info(f"Cleaned content: {cleaned_content}")

        # Prepare the cleaned post data to be inserted into the cleaned collection
        cleaned_post = {
            "video_id": video_id,  # The unique video ID
            "url": url,  # The URL of the YouTube video
            "cleaned_content": cleaned_content,  # The cleaned content
            # "original_content": content,  # Optionally store the original content
            "platform": "youtube",  # Platform is always 'youtube' in this case
        }

        # Insert the cleaned post into the 'rag_cleaned_data' collection
        cleaned_collection.insert_one(cleaned_post)

        logger.info(
            f"YouTube content cleaning and saving complete for video: {post['url']}"
        )

    # Cleaning function

    # GitHub Cleaner Function to fetch from MongoDB, clean, and store in cleaned collection
    def github_cleaner(self, post, cleaned_collection):
        """
        Connects to MongoDB, fetches GitHub repository data, cleans it, and stores it in 'rag_cleaned_data'.
        """

        # Extract necessary fields: URL and platform
        url = post.get("url", "")
        platform = post.get("platform", "github")

        # Combine all content values in the content object (ignore the keys)
        content_parts = []

        # Check if 'content' field exists in the post
        content = post.get("content", {})
        for value in content.values():
            if value:  # Only add non-empty values
                content_parts.append(value)  # Add the value to content_parts

        # Combine all content parts into a single string
        combined_content = " #### ".join(content_parts)

        # Clean the combined content
        cleaned_content = self.clean_text(combined_content)

        # Prepare the cleaned post data to be inserted into the cleaned collection
        cleaned_post = {
            "url": url,  # URL of the GitHub repository
            "cleaned_content": cleaned_content,  # Cleaned content
            "platform": platform,  # Platform (GitHub)
        }

        # Log the cleaned data to verify before saving
        # logger.info(f"Cleaned post to be inserted: {cleaned_post}")

        # Insert the cleaned post into the 'rag_cleaned_data' collection
        cleaned_collection.insert_one(cleaned_post)

        logger.info(
            f"GitHub content cleaning and saving complete for repo: {post['url']}"
        )


def etl_clean():

    task = Task.init(project_name="cs-gy-6613-rag", task_name="etl_cleaner")

    logger.info("Starting cleaning process of raw docs...")

    etl_clean_obj = ETL_Clean()

    mongo_client = get_mongo_client()
    db = mongo_client["rag"]  # Use the 'rag' database
    raw_data_collection = db["rag_raw_data"]  # Raw data collection
    cleaned_data_collection = db["rag_cleaned_data"]

    docs = raw_data_collection.find()

    for doc in docs:
        if doc["platform"] == "medium":
            etl_clean_obj.medium_cleaner(doc, cleaned_data_collection)

        elif doc["platform"] == "youtube":
            etl_clean_obj.youtube_cleaner(doc, cleaned_data_collection)

        elif doc["platform"] == "github":
            etl_clean_obj.github_cleaner(doc, cleaned_data_collection)

    logger.info("Finished cleaning.")

    task.close()


if __name__ == "__main__":
    etl_clean()

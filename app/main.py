from github import Github
from pymongo import MongoClient
from zenml.pipelines import pipeline
from zenml.steps import step

from configs import MONGO_CONNECTION_URL, GITHUB_TOKEN


# Step 1: Define GitHub File Tree Parser
def parse_repository_file_tree(repo_name, token):
    """Fetch and parse a GitHub repository file tree."""
    g = Github(token)
    repo = g.get_repo(repo_name)
    file_tree = repo.get_contents("")
    docs = []

    while file_tree:
        content = file_tree.pop(0)
        if content.type == "dir":
            file_tree.extend(repo.get_contents(content.path))
        elif (
            content.name.endswith((".md", ".rst", ".txt"))
            or "docs" in content.path.lower()
        ):
            docs.append(content.download_url)

    return docs


# Step 2: Define MongoDB Storage
def store_urls_in_mongodb(urls, db_name, collection_name):
    """Store documentation URLs in MongoDB."""
    client = MongoClient(MONGO_CONNECTION_URL)
    db = client[db_name]
    collection = db[collection_name]
    for url in urls:
        collection.insert_one({"url": url})
    client.close()


# Step 3: ZenML Steps
@step
def fetch_github_docs(repo_name: str, token: str) -> list:
    """ZenML step to fetch GitHub documentation."""
    urls = parse_repository_file_tree(repo_name, token)
    return urls


@step
def save_to_db(urls: list, db_name: str = "github_docs", collection_name: str = "ros2"):
    """ZenML step to save URLs to MongoDB."""
    store_urls_in_mongodb(urls, db_name, collection_name)


# Step 4: ZenML Pipeline
@pipeline
def github_etl_pipeline(fetch_docs, save_docs):
    """ZenML ETL pipeline."""
    urls = fetch_docs()
    save_docs(urls=urls)


# Step 5: Run the Pipeline
if __name__ == "__main__":
    # Fetch GitHub token from environment variable

    # Repository to parse
    repo_name = "ros/ros2_documentation"

    # Initialize and run the pipeline
    pipeline_instance = github_etl_pipeline(
        fetch_docs=fetch_github_docs(repo_name=repo_name, token=GITHUB_TOKEN),
        save_docs=save_to_db(),
    )
    pipeline_instance.run()

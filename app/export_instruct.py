from helpers.mongo_client import get_mongo_client

# export the instruct set to a file

mongo_client = get_mongo_client()
db = mongo_client["rag"]
collection = db["instruct_set"]

# Get all instruct set data
instruct_set_data = list(collection.find({}))

# Save instruct set data to hugging face dataset
import pandas as pd

instruct_set_df = pd.DataFrame(instruct_set_data)
instruct_set_df.to_csv("instruct_set.csv", index=False)


# upload this to hugging face dataset
from huggingface_hub import HfApi

# Initialize API
api = HfApi()

# Create a new dataset repository
repo_name = "cs-gy-6613-rag-instruct-set"
# api.create_repo(repo_id=f"your_username/{repo_name}", repo_type="dataset")

# Upload the file
api.upload_file(
    path_or_fileobj="instruct_set.csv",  # Or use json_path, parquet_path
    path_in_repo="dataset.csv",  # Desired name on the Hub
    repo_id=f"anindaghosh/{repo_name}",
    repo_type="dataset",
)

from clearml import Task, PipelineController
from pymongo import MongoClient


def extract_medium():
    # Example: Extract data from Medium
    # Implement your logic here

    # give me code to extract data from medium for certain topic

    return [{"source": "Medium", "topic": "ROS2", "content": "Sample content"}]


def transform_data(data):
    # Example: Transform data (cleaning, NLP, etc.)
    return [
        {"topic": item["topic"], "cleaned_content": item["content"]} for item in data
    ]


def load_to_mongodb(data):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["etl_pipeline"]
    collection = db["topics"]
    collection.insert_many(data)


def test():
    print("ok")


# Initialize ClearML pipeline
pipe = PipelineController(name="ETL Pipeline", project="Data Collection")

pipe.add_function_step("test", test)

# pipe.add_function_step("extract_medium", extract_medium)
# pipe.add_function_step(
#     "transform_data", transform_data, function_kwargs={"data": "${extract_medium}"}
# )
# pipe.add_function_step(
#     "load_to_mongodb", load_to_mongodb, function_kwargs={"data": "${transform_data}"}
# )

pipe.set_default_execution_queue(default_execution_queue="default")

# Starting the pipeline (in the background)
pipe.start()
# Wait until pipeline terminates
pipe.wait()
# cleanup everything
pipe.stop()

import pymongo
from app.configs import MONGO_CONNECTION_URL


def get_mongo_client():
    return pymongo.MongoClient(MONGO_CONNECTION_URL)

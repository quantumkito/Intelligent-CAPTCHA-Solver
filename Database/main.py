from dotenv import load_dotenv
import os
import pprint
from pymongo import MongoClient
import certifi

load_dotenv()
mongo_password = os.getenv("MONGO_PASSWORD")

client = MongoClient(f"mongodb+srv://username:{mongo_password}@cluster.mongodb.net/?retryWrites=true&w=majority", tlsCAFile=certifi.where())

dbs = client.list_database_names()
BreakingCaptcha_db = client.BreakingCaptcha
collections = BreakingCaptcha_db.list_collection_names()


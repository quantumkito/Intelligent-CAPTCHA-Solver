from dotenv import load_dotenv
import os
import pprint
from pymongo import MongoClient
import certifi

load_dotenv()
mongo_password = os.getenv("MONGO_PASSWORD")

client = MongoClient(f"mongodb+srv://username:{mongo_password}@cluster.mongodb.net/?retryWrites=true&w=majority", tlsCAFile=certifi.where())

BreakingCaptcha_db = client["BreakingCaptcha"]

def insert_user_doc(db, user_data):
    """
    Insert a user document into the MongoBD collection, checking for duplicate emails.
    Returns the inserted document ID or a message if the email already exists.
    """
    try:
        collection = db["User"]

        existing_user = collection.find_one({"Email": user_data["Email"]})
        if existing_user:
            return f"User with email {user_data['Email']} already exists!"
            
        inserted_id = collection.insert_one(user_data).inserted_id
        return f"User inserted with ID: {inserted_id}"

    except Exception as e:
        print(f"Error inserting document: {e}")
        return None
    
user_document = {
    "ID": "yyyyyyyyy",
    "FirstName": "Krtish",
    "LastName": "Vaidhyan",
    "Email": "krtish2008@gmail.com",
    "Username": "Kito",
    "Password": "****",
    "Date": "DD/MM/YYYY",
    "Data": "Multimedia_Data_ID"
}

result = insert_user_doc(BreakingCaptcha_db, user_document)
print(result)







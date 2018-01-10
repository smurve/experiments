from pymongo import MongoClient
client = MongoClient('mongodb://scylla:30017')
db = client.test_database
collection = db.test_collection
posts = db.posts

from pymongo import MongoClient
import pprint
client = MongoClient('mongodb://scylla:30017')
db = client.test_database
collection = db.test_collection
posts = db.posts
post={"author": "Wolfie", "title": "Athletic Professionalism"}
pos_id = posts.insert_one(post).inserted_id
pprint.pprint(pos_id)
pprint.pprint(posts.find_one())

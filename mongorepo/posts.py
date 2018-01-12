from pymongo import MongoClient


class PostsRepo:
    "Mongo-backed repository"

    def __init__(self, url):
        self.client = MongoClient(url)
        self.db = self.client.test_database
        self.posts = self.db.posts

    def find_one(self, query):
        return self.posts.find_one(query, {"_id": 0})

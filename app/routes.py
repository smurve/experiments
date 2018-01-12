from app import app
from mongorepo import posts
import json


@app.route('/')
@app.route('/index')
def index():
    return json.dumps(posts.find_one({}))

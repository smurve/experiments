from app import app
from mongorepo import posts


@app.route('/')
@app.route('/index')
def index():
    return posts.find_one()

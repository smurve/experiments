from __future__ import print_function
from app import app
from mongorepo import posts
from flask import render_template, send_from_directory
import json

user = {'username': 'Wolfie'}


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Ellie', user=user)


@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)


@app.route('/reload')
def reload():
    load_images()
    return render_template('index.html', title='Ellie', user=user)


@app.route('/mongo')
def mongo():
    return json.dumps(posts.find_one({}))


def load_images():
    print("loading...")

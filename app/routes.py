from __future__ import print_function
from app import app
from mongorepo import posts
from flask import render_template, send_from_directory
import json
from services.predictions.mnist_prediction import MnistPredictionService

user = {'username': 'Wolfie'}

mnist_service = MnistPredictionService()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Ellie', user=user)


@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)


@app.route('/failed-images')
@app.route('/reload')
def reload():
    width = 18
    failed = load_images()
    grid = [failed[rownum * width: (rownum + 1) * width]
            for rownum in range(int(len(failed) / width) + 1)]

    return render_template('failed-images.html',
                           title='Ellie', user=user,
                           image_grid=grid)


@app.route('/mongo')
def mongo():
    return json.dumps(posts.find_one({}))


def load_images():
    print("loading...")
    return mnist_service.create_failed_pngs(10000)

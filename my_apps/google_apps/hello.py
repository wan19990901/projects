import os
import pymysql
from flask import (Flask, request, session, g, redirect, url_for, abort,
render_template, flash)

app = Flask(__name__)
app.config.from_object(__name__) # load config from this file , flaskr.py
# Load default config and override config from an environment variable
app.config.update(
DATABASE=os.path.join(app.root_path, 'flaskr.db'),
SECRET_KEY=b'_5#y2L"F4Q8z\n\xec]/',
USERNAME='guangya',
PASSWORD='password'
)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/hello')
def hello():
    return 'Hello, World'
@app.route('/user/<username>')
def show_user_profile(username):
# show the user profile for that user
    return 'User %s' % username
@app.route('/post/<int:post_id>')
def show_post(post_id):
# show the post with the given id, the id is an integer
    return 'Post %d' % post_id
@app.route('/path/<path:subpath>')
@app.route('/login')
def login():
    return 'login'
def show_subpath(subpath):
# show the subpath after /path/
    return 'Subpath %s' % subpath
@app.route('/user/<username>')
def profile(username):
    return '{}s profile'.format(username)
# with app.test_request_context():
#     print(url_for('index'))
#     print(url_for('login'))
#     print(url_for('login', next='/'))
#     print(url_for('profile', username='John Doe'))
# /
# /login
# /login?next=/
# /user/John%20Doe
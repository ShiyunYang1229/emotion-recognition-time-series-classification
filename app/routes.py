from flask import render_template, flash, redirect, url_for, request, send_from_directory, jsonify
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import User
from time import gmtime, strftime
import json
from EmotionML import EmotionML
from Mindwave import Mindwave
import time, datetime
from Dynamo import Dynamo


@app.route('/assets/<path:path>')
def static_file(path):
    return send_from_directory('static/assets/', path)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route("/introduction")
def introduction():
    return render_template('introduction.html')


@app.route("/test")
@login_required
def test():
    return render_template('test_ajax.html')


# data collecting and uploading in /test @button
def func1():
    sensor = Mindwave()
    json_data = sensor.collect_data()
    if json_data:
        ML = EmotionML()
        ML.load_data(json_data)
        ML.preprocess()
        res = ML.predict()

        dynamo = Dynamo()
        username = current_user.username
        now = strftime("%a, %d %b %Y %X GMT", gmtime())
        data = json.dumps(res)
        result = str(res["vote0"][0])
        res['time'] = now
        dynamo.dynamoAdd(username, now, data, result)
        print(username, now, data, result)
        return res
    return {}


def getKey(item):
    return time.mktime(time.strptime(item['time'], '%a, %d %b %Y %X GMT'))


# history data retrieving from dynamoDB with username provided for /profile
def func2(username):
    dynamo = Dynamo()
    data = dynamo.dynamoQuery(username)
    pie =[0, 0, 0, 0, 0]
    data = sorted(data, key=getKey)
    print("data",data)

    for d in data:
        print(d)
        pie[json.loads(d['data'])['vote0'][1]-1] += 1

    if not data:
        return {}
    res = {"earliest": data[0]["time"],
           "latest": data[-1]["time"],
           "num": len(data),
           "data": data,
           "pie": pie
           }

    return res


@app.route('/collect', methods=['GET'])
@login_required
def collect():
    res = func1()
    if res:
        res["name"] = current_user.username
        return jsonify(res)
    return "", 500


@app.route('/result', methods=['POST'])
@login_required
def result():
    res = json.loads(request.form['var_res'])
    return render_template('result.html', data=res)


@app.route("/therapy")
@login_required
def therapy():
    state=request.args.get('state')
    return render_template('therapy.html', state=state)


@app.route("/fun-clips")
def fun_clips():
    return render_template('fun-clips.html')


@app.route("/sad-scenes")
def sad_scenes():
    return render_template('sad-scenes.html')


@app.route('/profile')
@login_required
def profile():
    res = func2(current_user.username)
    if not res:
        return redirect(url_for('test'))
    return render_template('profile_extend.html', res=res)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

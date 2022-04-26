# save this as app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("login.html")

app.run(host="localhost", port=9286, debug=True)
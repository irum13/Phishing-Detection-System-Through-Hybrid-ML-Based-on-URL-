import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from flask import Flask, redirect, render_template, request, url_for

from feature import FeatureExtraction


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "signup.db"
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="/static")
gbc = None


def get_model():
    global gbc
    if gbc is not None:
        return gbc

    with MODEL_PATH.open("rb") as file:
        gbc = pickle.load(file)
    return gbc


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS info (
                user TEXT PRIMARY KEY,
                email TEXT,
                password TEXT NOT NULL,
                mobile TEXT,
                name TEXT
            )
            """
        )


def normalize_url(raw_url):
    raw_url = raw_url.strip()
    if raw_url and not raw_url.startswith(("http://", "https://")):
        return f"https://{raw_url}"
    return raw_url


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/index")
@app.route("/url", methods=["GET", "POST"])
def url():
    if request.method == "POST":
        submitted_url = normalize_url(request.form.get("url", ""))
        if not submitted_url:
            return render_template("index.html", error="Please enter a URL.", xx=-1)

        try:
            model = get_model()
            obj = FeatureExtraction(submitted_url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)
            y_pred = model.predict(x)[0]
            probabilities = model.predict_proba(x)[0]
        except Exception as exc:
            return render_template(
                "index.html",
                error=f"Unable to analyze this URL: {exc}",
                xx=-1,
            )

        phishing_probability = float(probabilities[0])
        safe_probability = float(probabilities[1])
        is_safe = int(y_pred) == 1

        return render_template(
            "result.html",
            url=submitted_url,
            xx=round(safe_probability, 2),
            safe_percent=round(safe_probability * 100, 2),
            phishing_percent=round(phishing_probability * 100, 2),
            is_safe=is_safe,
        )

    return render_template("index.html", xx=-1)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["POST"])
def contact():
    return "Thanks, your message was received."


@app.route("/logon")
def logon():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("signin.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET" and not request.args:
        return render_template("signup.html")

    user = request.values.get("user", "").strip()
    name = request.values.get("name", "").strip()
    email = request.values.get("email", "").strip()
    mobile = request.values.get("mobile", "").strip()
    password = request.values.get("password", "")

    if not user or not password:
        return render_template("signup.html", error="Username and password are required.")

    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO info (user, email, password, mobile, name) VALUES (?, ?, ?, ?, ?)",
                (user, email, password, mobile, name),
            )
    except sqlite3.IntegrityError:
        return render_template("signup.html", error="That username already exists.")

    return render_template("signin.html", message="Account created. Please sign in.")


@app.route("/predict1", methods=["POST"])
def predict1():
    # Backward-compatible endpoint for the old OTP page. The local app now signs up directly.
    return redirect(url_for("login"))


@app.route("/signin", methods=["GET", "POST"])
def signin():
    user = request.values.get("user", "").strip()
    password = request.values.get("password", "")

    if not user or not password:
        return render_template("signin.html")

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT user, password FROM info WHERE user = ? AND password = ?",
            (user, password),
        )
        data = cur.fetchone()

    if data:
        return render_template("index.html", user=user, xx=-1)

    return render_template("signin.html", error="Invalid username or password.")


@app.route("/notebook")
def notebook():
    return render_template("notebook.html")


@app.route("/val")
def val():
    return render_template("val.html")


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET","POST"])
def home():
    prediction=None
    if request.method=="POST":
        text=request.form["email"]
        vec=vectorizer.transform([text])
        prediction=model.predict(vec)[0]
    return render_template("index.html",prediction=prediction)

app.run(host="0.0.0.0",port=5000)

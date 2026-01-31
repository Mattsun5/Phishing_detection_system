from flask import Flask, render_template, request
import joblib, json, os

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load metrics
with open("metrics.json") as f:
    metrics = json.load(f)

@app.route("/", methods=["GET","POST"])
def home():
    prediction=None
    if request.method=="POST":
        text=request.form["email"]
        vec=vectorizer.transform([text])
        prediction=model.predict(vec)[0]
    return render_template(
        "index.html",
        prediction=prediction,
        metrics=metrics
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

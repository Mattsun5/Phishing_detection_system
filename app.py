from flask import Flask, render_template, request
import joblib, json, os

app = Flask(__name__)

# Load Model and Vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load metrics
with open("metrics.json") as f:
    # We load the entire data object (which contains both 'metrics' and 'confusion')
    report_data = json.load(f)

@app.route("/", methods=["GET","POST"])
def home():
    prediction=None
    if request.method == "POST":
        try:
            text = request.form["email"]
            if text.strip(): # Ensure text is not empty
                vec = vectorizer.transform([text])
                prediction = model.predict(vec)[0]
        except Exception as e:
            print(f"Prediction Error: {e}")

    return render_template(
        "index.html",
        prediction=prediction,
        # Access the 'metrics' key from the JSON to pass the arrays ([0.9, 0.8...])
        metrics=report_data["metrics"],
        # Access the 'confusion' key from the JSON to pass the matrix ([[TN, FP], ...])
        confusion=report_data["confusion"]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import json
import joblib

# 1. Load Data
try:
    df = pd.read_csv("data/emails.csv")
    print(f"Data loaded: {len(df)} records")
except FileNotFoundError:
    print("Error: 'data/emails.csv' not found. Please ensure the data file exists.")
    exit()

X = df["text"]
y = df["label"]

# 2. Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42
)

# 3. Define Models to Match Chart Labels
# Order matters: ["Naive Bayes", "SVM", "Random Forest"]
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'), # Linear kernel is often faster/better for text
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
# Storage for metrics to match Chart.js structure
results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

final_confusion_matrix = []

print("Training models...")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)

    # Define which string label is the "positive" one (the one you are trying to detect)
    target_label = "phishing"
    
    # Calculate Metrics (rounded for cleaner JSON)
    results["accuracy"].append(round(accuracy_score(y_test, y_pred), 3))
    results["precision"].append(round(precision_score(y_test, y_pred, pos_label=target_label, zero_division=0), 3))
    results["recall"].append(round(recall_score(y_test, y_pred, pos_label=target_label, zero_division=0), 3))
    results["f1"].append(round(f1_score(y_test, y_pred, pos_label=target_label, zero_division=0), 3))

    # If this is the Random Forest model, save its artifacts for Production
    if name == "Random Forest":
        # Save model and vectorizer
        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        
        # Capture Confusion Matrix for the Heatmap
        # Returns [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_test, y_pred)
        final_confusion_matrix = cm.tolist() 

# 4. Construct Final JSON Structure
output_data = {
    "metrics": results,              # Arrays for the Bar Chart
    "confusion": final_confusion_matrix # 2D Array for the Heatmap
}

# 5. Save to JSON
with open("metrics.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Training complete.")
print(f"Metrics saved to metrics.json: {json.dumps(output_data, indent=2)}")

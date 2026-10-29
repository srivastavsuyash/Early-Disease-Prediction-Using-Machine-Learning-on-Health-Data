import os

import json
import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.joblib")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "models", "symptom_list.json")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

with open(SYMPTOMS_PATH, "r", encoding="utf-8") as f:
    symptoms = json.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Disease Risk Predictor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {
    font-family: Arial, sans-serif;
    background: #f3f5fb;
    margin: 0;
    padding: 30px 15px;
    color: #20243a;
}
.container {
    max-width: 760px;
    margin: auto;
    background: white;
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 6px 24px #18204415;
}
h1 { color: #4f46e5; }
p { line-height: 1.6; }
select, button {
    width: 100%;
    padding: 12px;
    margin-top: 12px;
    border-radius: 8px;
    font-size: 16px;
}
select { border: 1px solid #ccd0df; }
button {
    background: #4f46e5;
    color: white;
    border: 0;
    cursor: pointer;
}
.result {
    background: #eef2ff;
    padding: 16px;
    border-radius: 10px;
    margin-top: 20px;
}
.warning {
    color: #8a4b08;
    background: #fff7e6;
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
}
</style>
</head>
<body>
<div class="container">
    <h1>Disease Risk Predictor</h1>
    <p>Select symptoms to view the model's predicted disease category.</p>
    <form method="POST">
        <label for="symptoms"><b>Select symptoms</b></label>
        <select name="symptoms" id="symptoms" multiple size="12">
        {% for symptom in all_symptoms %}
            <option value="{{ symptom }}"
            {% if symptom in selected %}selected{% endif %}>
                {{ symptom.replace('_', ' ').title() }}
            </option>
        {% endfor %}
        </select>
        <p>Tip: Hold Ctrl (Windows) while clicking to select multiple symptoms.</p>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
    <div class="result">
        <h2>Model Prediction</h2>
        <p><b>Predicted category:</b> {{ prediction }}</p>
        <p>This is a machine-learning output, not a diagnosis.</p>
    </div>
    {% endif %}

    <div class="warning">
        Educational demonstration only. This tool is not medically validated
        and must not be used to diagnose or treat any condition.
        Consult a qualified healthcare professional for health concerns.
    </div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    selected = request.form.getlist("symptoms") if request.method == "POST" else []
    prediction = None

    if selected:
        # Build a binary feature row in the order expected by the model.
        feature_row = {symptom: int(symptom in selected) for symptom in symptoms}

        # Use a DataFrame so models trained with feature names receive them.
        X = pd.DataFrame([feature_row], columns=symptoms)
        predicted_value = model.predict(X)[0]

        try:
            prediction = str(encoder.inverse_transform([predicted_value])[0])
        except (ValueError, TypeError):
            prediction = str(predicted_value)

    return render_template_string(
        HTML,
        all_symptoms=symptoms,
        selected=selected,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

# 🩺 Early Disease Prediction Using Machine Learning

A machine learning-based project that predicts possible disease categories from selected symptoms. It combines Python, machine learning, and a Flask web application to demonstrate symptom-based prediction and explainable AI concepts.

## 🚀 Features

* Symptom-based disease prediction
* Trained machine learning model
* Disease label encoding and decoding
* Model evaluation using Accuracy, Macro F1-score, and Top-3 evaluation
* Data visualization and confusion matrix analysis
* SHAP-based feature importance analysis
* Flask-based web interface

## 🛠️ Tech Stack

* **Language:** Python
* **ML:** Scikit-learn, NumPy, Pandas
* **Model Storage:** Joblib
* **Web Framework:** Flask
* **Explainable AI:** SHAP
* **Deployment:** Render (planned)

## 📂 Project Files

* `app.py` — Flask web application
* `requirements.txt` — Project dependencies
* `best_model.joblib` — Trained machine learning model
* `label_encoder.joblib` — Encoded disease labels
* `symptom_list.json` — Supported symptoms
* `*.png` — Model evaluation and analysis visualizations

*File locations may vary depending on the repository structure.*

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/srivastavsyuyash/Early-Disease-Prediction-Using-Machine-Learning-on-Health-Data.git
cd Early-Disease-Prediction-Using-Machine-Learning-on-Health-Data
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser if the application starts on port 5000.

## 🔬 Future Improvements

* Improve prediction validation and model reliability
* Add automated testing
* Enhance the user interface
* Deploy the application online
* Improve prediction explanations

## ⚠️ Disclaimer

This project is developed for educational purposes only. It is not a clinically validated diagnostic tool and must not be used as a substitute for professional medical advice.

## 👨‍💻 Author

**Suyash Srivastava**

GitHub: [@srivastavsuyash](https://github.com/srivastavsyuyash)


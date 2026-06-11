import os
import json
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'prediction_history.json')


def load_model(name):
    model = joblib.load(os.path.join(MODELS_DIR, f'{name}_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'{name}_scaler.pkl'))
    return model, scaler


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_accuracy(name):
    acc = {
        'diabetes': 94.00,
        'heart': 81.50,
        'breast_cancer': 98.25
    }
    return acc.get(name, 0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    prediction = None
    if request.method == 'POST':
        try:
            gender = request.form.get('gender', 'male')
            pregnancies = float(request.form.get('pregnancies', 0))
            features = [pregnancies, float(request.form['glucose']),
                float(request.form['blood_pressure']), float(request.form['skin_thickness']),
                float(request.form['insulin']), float(request.form['bmi']),
                float(request.form['dpf']), float(request.form['age'])]
            model, scaler = load_model('diabetes')
            scaled = scaler.transform([features])
            result = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0]
            prediction = {
                'result': 'Positive' if result == 1 else 'Negative',
                'confidence': f"{max(probability) * 100:.1f}",
                'is_positive': bool(result)
            }
            history = load_history()
            history.append({
                'disease': 'Diabetes',
                'result': prediction['result'],
                'confidence': prediction['confidence'],
                'date': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            save_history(history)
        except Exception as e:
            prediction = {'error': str(e)}
    return render_template('diabetes.html', prediction=prediction, accuracy=get_accuracy('diabetes'))


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction = None
    if request.method == 'POST':
        try:
            features = [float(request.form[f]) for f in [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]]
            model, scaler = load_model('heart')
            scaled = scaler.transform([features])
            result = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0]
            prediction = {
                'result': 'Positive' if result == 1 else 'Negative',
                'confidence': f"{max(probability) * 100:.1f}",
                'is_positive': bool(result)
            }
            history = load_history()
            history.append({
                'disease': 'Heart Disease',
                'result': prediction['result'],
                'confidence': prediction['confidence'],
                'date': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            save_history(history)
        except Exception as e:
            prediction = {'error': str(e)}
    return render_template('heart.html', prediction=prediction, accuracy=get_accuracy('heart'))


@app.route('/breast-cancer', methods=['GET', 'POST'])
def breast_cancer():
    prediction = None
    if request.method == 'POST':
        try:
            features = [float(request.form[f]) for f in request.form.keys()]
            model, scaler = load_model('breast_cancer')
            scaled = scaler.transform([features])
            result = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0]
            prediction = {
                'result': 'Benign' if result == 1 else 'Malignant',
                'confidence': f"{max(probability) * 100:.1f}",
                'is_positive': bool(result == 1)
            }
            history = load_history()
            history.append({
                'disease': 'Breast Cancer',
                'result': prediction['result'],
                'confidence': prediction['confidence'],
                'date': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            save_history(history)
        except Exception as e:
            prediction = {'error': str(e)}
    return render_template('breast_cancer.html', prediction=prediction, accuracy=get_accuracy('breast_cancer'))


@app.route('/dashboard')
def dashboard():
    history = load_history()
    stats = {
        'total': len(history),
        'diabetes': sum(1 for h in history if h['disease'] == 'Diabetes'),
        'heart': sum(1 for h in history if h['disease'] == 'Heart Disease'),
        'breast_cancer': sum(1 for h in history if h['disease'] == 'Breast Cancer'),
        'positive': sum(1 for h in history if h['result'] in ['Positive', 'Malignant']),
        'negative': sum(1 for h in history if h['result'] in ['Negative', 'Benign']),
    }
    return render_template('dashboard.html', history=history, stats=stats)


@app.route('/api/stats')
def api_stats():
    history = load_history()
    stats = {
        'total': len(history),
        'disease_counts': {},
        'result_counts': {'Positive': 0, 'Negative': 0},
        'recent': history[-10:] if history else []
    }
    for h in history:
        d = h['disease']
        stats['disease_counts'][d] = stats['disease_counts'].get(d, 0) + 1
        if h['result'] in ['Positive', 'Malignant']:
            stats['result_counts']['Positive'] += 1
        else:
            stats['result_counts']['Negative'] += 1
    return jsonify(stats)


@app.route('/clear-history', methods=['POST'])
def clear_history():
    save_history([])
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

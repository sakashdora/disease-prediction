import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def generate_diabetes_data(n_samples=2000):
    np.random.seed(42)
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 32, n_samples).clip(44, 200),
        'BloodPressure': np.random.normal(69, 19, n_samples).clip(24, 122),
        'SkinThickness': np.random.normal(20, 16, n_samples).clip(7, 99),
        'Insulin': np.random.normal(80, 115, n_samples).clip(14, 846),
        'BMI': np.random.normal(32, 8, n_samples).clip(18, 67),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
    }
    df = pd.DataFrame(data)
    risk = (
        (df['Glucose'] > 140).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) * 2 +
        (df['Age'] > 45).astype(int) +
        (df['BloodPressure'] > 80).astype(int) +
        (df['Insulin'] > 130).astype(int) +
        (df['Pregnancies'] > 6).astype(int)
    )
    df['Outcome'] = (risk + np.random.binomial(1, 0.15, n_samples) > 3).astype(int)
    return df


def generate_heart_data(n_samples=2000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.normal(131, 17, n_samples).clip(94, 200).astype(int),
        'chol': np.random.normal(246, 51, n_samples).clip(126, 564).astype(int),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.normal(149, 22, n_samples).clip(71, 202).astype(int),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 5, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
    }
    df = pd.DataFrame(data)
    risk = (
        (df['age'] > 55).astype(int) * 2 +
        (df['cp'] > 1).astype(int) * 2 +
        (df['thalach'] < 130).astype(int) +
        (df['oldpeak'] > 2).astype(int) * 2 +
        (df['exang'] == 1).astype(int) +
        (df['chol'] > 280).astype(int) +
        (df['trestbps'] > 140).astype(int)
    )
    df['target'] = (risk + np.random.binomial(1, 0.15, n_samples) > 3).astype(int)
    return df


def train_diabetes_model():
    print("Training Diabetes Prediction Model...")
    df = generate_diabetes_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, os.path.join('models', 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join('models', 'diabetes_scaler.pkl'))
    return accuracy


def train_heart_model():
    print("Training Heart Disease Prediction Model...")
    df = generate_heart_data()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, os.path.join('models', 'heart_model.pkl'))
    joblib.dump(scaler, os.path.join('models', 'heart_scaler.pkl'))
    return accuracy


def train_breast_cancer_model():
    print("Training Breast Cancer Prediction Model...")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, os.path.join('models', 'breast_cancer_model.pkl'))
    joblib.dump(scaler, os.path.join('models', 'breast_cancer_scaler.pkl'))
    return accuracy


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    print("=" * 50)
    print("  DISEASE PREDICTION MODEL TRAINING")
    print("=" * 50)
    accuracies = {}
    accuracies['diabetes'] = train_diabetes_model()
    accuracies['heart'] = train_heart_model()
    accuracies['breast_cancer'] = train_breast_cancer_model()
    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 50)
    for disease, acc in accuracies.items():
        print(f"  {disease.title():15s}: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nModels saved to 'models/' directory.")

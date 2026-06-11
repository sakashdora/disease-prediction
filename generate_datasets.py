import numpy as np
import pandas as pd
import os


def generate_diabetes_csv():
    np.random.seed(42)
    n = 2000
    data = {
        'Pregnancies': np.random.randint(0, 17, n),
        'Glucose': np.random.normal(120, 32, n).clip(44, 200).astype(int),
        'BloodPressure': np.random.normal(69, 19, n).clip(24, 122).astype(int),
        'SkinThickness': np.random.normal(20, 16, n).clip(7, 99).astype(int),
        'Insulin': np.random.normal(80, 115, n).clip(14, 846).astype(int),
        'BMI': np.random.normal(32, 8, n).clip(18, 67).round(1),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n).round(3),
        'Age': np.random.randint(21, 81, n),
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
    df['Outcome'] = (risk + np.random.binomial(1, 0.15, n) > 3).astype(int)
    df.to_csv('diabetes.csv', index=False)
    print(f"diabetes.csv created: {len(df)} rows")


def generate_heart_csv():
    np.random.seed(42)
    n = 2000
    data = {
        'age': np.random.randint(29, 77, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.normal(131, 17, n).clip(94, 200).astype(int),
        'chol': np.random.normal(246, 51, n).clip(126, 564).astype(int),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.normal(149, 22, n).clip(71, 202).astype(int),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.random.uniform(0, 6.2, n).round(1),
        'slope': np.random.randint(0, 3, n),
        'ca': np.random.randint(0, 5, n),
        'thal': np.random.randint(0, 4, n),
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
    df['target'] = (risk + np.random.binomial(1, 0.15, n) > 3).astype(int)
    df.to_csv('heart.csv', index=False)
    print(f"heart.csv created: {len(df)} rows")


def generate_breast_cancer_csv():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('breast_cancer.csv', index=False)
    print(f"breast_cancer.csv created: {len(df)} rows")


if __name__ == '__main__':
    os.makedirs('datasets', exist_ok=True)
    os.chdir('datasets')
    print("Generating CSV datasets...")
    generate_diabetes_csv()
    generate_heart_csv()
    generate_breast_cancer_csv()
    print("All datasets created in datasets/ folder")

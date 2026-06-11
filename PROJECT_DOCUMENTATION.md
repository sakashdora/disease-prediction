# Disease Prediction System - Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Technology Stack](#technology-stack)
4. [Data Collection](#data-collection)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [End-to-End Workflow](#end-to-end-workflow)
8. [API Endpoints](#api-endpoints)
9. [Screenshots & Features](#screenshots--features)
10. [Future Enhancements](#future-enhancements)
11. [References](#references)

---

## Project Overview

**Project Name**: Disease Prediction System  
**Created By**: S AKASH DORA  
**Course**: MCA Project 2026  
**Type**: Machine Learning Web Application  

### Problem Statement
Early detection of diseases like Diabetes, Heart Disease, and Breast Cancer can save lives. This system uses machine learning algorithms to predict the risk of these diseases based on patient health parameters.

### Objectives
- Build ML models to predict 3 diseases
- Create a user-friendly web interface
- Provide instant predictions with confidence scores
- Store prediction history for tracking

---

## Project Structure

```
disease-prediction/
│
├── app.py                      # Main Flask application (backend)
├── train_models.py             # ML model training script
├── generate_datasets.py        # Dataset generation script
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── PROJECT_DOCUMENTATION.md    # This file
├── .gitignore                  # Git ignore rules
│
├── models/                     # Trained ML models
│   ├── diabetes_model.pkl      # Random Forest model
│   ├── diabetes_scaler.pkl     # StandardScaler for diabetes
│   ├── heart_model.pkl         # Logistic Regression model
│   ├── heart_scaler.pkl        # StandardScaler for heart
│   ├── breast_cancer_model.pkl # SVM model
│   └── breast_cancer_scaler.pkl# StandardScaler for breast cancer
│
├── datasets/                   # Training datasets (CSV)
│   ├── diabetes.csv            # 2000 rows, 9 columns
│   ├── heart.csv               # 2000 rows, 14 columns
│   └── breast_cancer.csv       # 569 rows, 31 columns
│
├── templates/                  # HTML templates
│   ├── index.html              # Home page
│   ├── diabetes.html           # Diabetes prediction form
│   ├── heart.html              # Heart disease prediction form
│   ├── breast_cancer.html      # Breast cancer prediction form
│   └── dashboard.html          # Analytics dashboard
│
└── static/                     # Static files
    └── css/
        └── style.css           # Custom CSS styles
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | Python 3.11 | Core programming language |
| **Web Framework** | Flask 3.0 | HTTP server and routing |
| **ML Library** | scikit-learn 1.3 | Model training and prediction |
| **Data Processing** | pandas 2.1 | Data manipulation |
| **Numerical Computing** | numpy 1.26 | Array operations |
| **Model Serialization** | joblib | Save/load trained models |
| **Frontend** | HTML5, CSS3 | User interface |
| **CSS Framework** | Bootstrap 5.3 | Responsive design |
| **Charts** | Chart.js | Data visualization |

---

## Data Collection

### 1. Diabetes Dataset

| Property | Details |
|----------|---------|
| **Source** | Pima Indians Diabetes Database (UCI Repository) |
| **Original Samples** | 768 |
| **Generated Samples** | 2000 |
| **Features** | 8 |
| **Target** | Binary (0=No Diabetes, 1=Diabetes) |
| **File** | `datasets/diabetes.csv` |

**Features Explained**:

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-17 |
| Glucose | Plasma glucose concentration (mg/dL) | 44-200 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 24-122 |
| SkinThickness | Triceps skin fold thickness (mm) | 7-99 |
| Insulin | 2-Hour serum insulin (mu U/ml) | 14-846 |
| BMI | Body mass index | 18-67 |
| DiabetesPedigreeFunction | Family history score | 0.078-2.42 |
| Age | Age in years | 21-81 |

### 2. Heart Disease Dataset

| Property | Details |
|----------|---------|
| **Source** | UCI Heart Disease Dataset (Cleveland) |
| **Original Samples** | 303 |
| **Generated Samples** | 2000 |
| **Features** | 13 |
| **Target** | Binary (0=No Disease, 1=Heart Disease) |
| **File** | `datasets/heart.csv` |

**Features Explained**:

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age in years | 29-77 |
| sex | Gender (1=Male, 0=Female) | 0-1 |
| cp | Chest Pain Type (0-3) | 0-3 |
| trestbps | Resting Blood Pressure (mm Hg) | 94-200 |
| chol | Serum Cholesterol (mg/dl) | 126-564 |
| fbs | Fasting Blood Sugar > 120 (1=True) | 0-1 |
| restecg | Resting ECG results (0-2) | 0-2 |
| thalach | Maximum Heart Rate | 71-202 |
| exang | Exercise Induced Angina (1=Yes) | 0-1 |
| oldpeak | ST Depression | 0-6.2 |
| slope | Slope of Peak Exercise ST | 0-2 |
| ca | Number of Major Vessels | 0-3 |
| thal | Thalassemia Type | 0-3 |

### 3. Breast Cancer Dataset

| Property | Details |
|----------|---------|
| **Source** | Wisconsin Breast Cancer (sklearn built-in) |
| **Samples** | 569 |
| **Features** | 30 |
| **Target** | Binary (0=Malignant, 1=Benign) |
| **File** | `datasets/breast_cancer.csv` |

**Features Explained** (10 attributes x 3 measurements each):

| Attribute | Mean | SE | Worst |
|-----------|------|-----|-------|
| radius | mean_radius | se_radius | worst_radius |
| texture | mean_texture | se_texture | worst_texture |
| perimeter | mean_perimeter | se_perimeter | worst_perimeter |
| area | mean_area | se_area | worst_area |
| smoothness | mean_smoothness | se_smoothness | worst_smoothness |
| compactness | mean_compactness | se_compactness | worst_compactness |
| concavity | mean_concavity | se_concavity | worst_concavity |
| concave_points | mean_concave_points | se_concave_points | worst_concave_points |
| symmetry | mean_symmetry | se_symmetry | worst_symmetry |
| fractal_dimension | mean_fractal_dimension | se_fractal_dimension | worst_fractal_dimension |

---

## Data Preprocessing

### Steps Applied

1. **Data Cleaning**
   - No missing values in generated datasets
   - No duplicate rows

2. **Feature Scaling**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

3. **Train-Test Split**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```
   - Training: 80%
   - Testing: 20%

---

## Model Training

### 1. Diabetes Model

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Classifier |
| **n_estimators** | 100 |
| **random_state** | 42 |
| **Accuracy** | 94.00% |
| **Model File** | `models/diabetes_model.pkl` |
| **Scaler File** | `models/diabetes_scaler.pkl` |

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
```

### 2. Heart Disease Model

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression |
| **max_iter** | 1000 |
| **random_state** | 42 |
| **Accuracy** | 81.50% |
| **Model File** | `models/heart_model.pkl` |
| **Scaler File** | `models/heart_scaler.pkl` |

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
```

### 3. Breast Cancer Model

| Property | Value |
|----------|-------|
| **Algorithm** | Support Vector Machine (SVM) |
| **kernel** | rbf |
| **probability** | True |
| **random_state** | 42 |
| **Accuracy** | 98.25% |
| **Model File** | `models/breast_cancer_model.pkl` |
| **Scaler File** | `models/breast_cancer_scaler.pkl` |

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
```

---

## End-to-End Workflow

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Diabetes  │  │    Heart    │  │   Breast    │            │
│  │   Form      │  │   Form      │  │   Cancer    │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK BACKEND (app.py)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Receive form data                                   │   │
│  │  2. Convert to feature array                            │   │
│  │  3. Load appropriate model + scaler                     │   │
│  │  4. Scale input features                                │   │
│  │  5. Make prediction                                     │   │
│  │  6. Get confidence score                                │   │
│  │  7. Save to history                                     │   │
│  │  8. Return result to template                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML MODELS (models/)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Random    │  │  Logistic   │  │     SVM     │            │
│  │   Forest    │  │ Regression  │  │             │            │
│  │  (94.00%)   │  │  (81.50%)   │  │  (98.25%)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION RESULT                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Result: Positive/Negative/Benign/Malignant            │   │
│  │  Confidence: XX.X%                                      │   │
│  │  Saved to: prediction_history.json                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **User Opens App**
   - Flask serves `index.html`
   - User sees 3 disease options

2. **User Selects Disease**
   - Clicks on Diabetes/Heart/Breast Cancer
   - Flask serves the corresponding form

3. **User Fills Form**
   - Enters health parameters
   - For diabetes: selects gender (pregnancy field hidden for males)

4. **Form Submission**
   - POST request to Flask backend
   - Data extracted from form fields

5. **Model Loading**
   ```python
   model = joblib.load('models/diabetes_model.pkl')
   scaler = joblib.load('models/diabetes_scaler.pkl')
   ```

6. **Feature Scaling**
   ```python
   scaled_features = scaler.transform([features])
   ```

7. **Prediction**
   ```python
   prediction = model.predict(scaled_features)
   probability = model.predict_proba(scaled_features)
   ```

8. **Result Display**
   - Result shown with color coding
   - Confidence percentage displayed
   - Saved to history file

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET/POST | `/diabetes` | Diabetes prediction |
| GET/POST | `/heart` | Heart disease prediction |
| GET/POST | `/breast-cancer` | Breast cancer prediction |
| GET | `/dashboard` | Analytics dashboard |
| GET | `/api/stats` | JSON statistics API |
| POST | `/clear-history` | Clear prediction history |

---

## Screenshots & Features

### Home Page
- Hero section with project title
- 3 disease cards with accuracy badges
- "How It Works" section
- Footer with creator name

### Prediction Forms
- Clean, responsive forms
- Input validation
- Real-time predictions
- Color-coded results (Green=Safe, Red=Risk)

### Dashboard
- Total predictions count
- Predictions by disease (bar chart)
- Results distribution (doughnut chart)
- Prediction history table
- Clear history button

---

## Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| User Authentication | Login system for tracking individual history |
| Database | SQLite/PostgreSQL instead of JSON file |
| More Diseases | Add Parkinson's, Liver Disease, etc. |
| API Deployment | Deploy as REST API for mobile apps |
| Model Improvement | Use deep learning models |
| PDF Reports | Generate downloadable prediction reports |
| Email Alerts | Send risk alerts via email |
| Multi-language | Support for regional languages |

---

## References

1. UCI Machine Learning Repository - https://archive.ics.uci.edu/
2. scikit-learn Documentation - https://scikit-learn.org/
3. Flask Documentation - https://flask.palletsprojects.com/
4. Bootstrap Documentation - https://getbootstrap.com/
5. Chart.js Documentation - https://www.chartjs.org/

---

## Author

**S AKASH DORA**  
MCA Student  
Project Year: 2026

---

## License

This project is for educational and academic purposes only.

---

**Disclaimer**: This system is for educational purposes only. Predictions may not be accurate. Always consult an experienced doctor for proper medical diagnosis and treatment.

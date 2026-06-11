# Disease Prediction System

A machine learning-based web application that predicts the risk of **Diabetes**, **Heart Disease**, and **Breast Cancer** using trained ML models. Built with Python, Flask, and scikit-learn.

---

## Features

- **Diabetes Prediction** - Predicts diabetes risk using glucose level, BMI, age, and other health parameters
- **Heart Disease Prediction** - Assesses heart disease risk using cholesterol, blood pressure, ECG results, and more
- **Breast Cancer Classification** - Classifies breast cancer as benign or malignant using cell nucleus characteristics
- **Analytics Dashboard** - Visualizes prediction history with charts and statistics
- **Prediction History** - Stores all predictions for later review
- **Gender-Aware Forms** - Diabetes form adapts based on gender (pregnancy field only for females)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| Machine Learning | scikit-learn, pandas, numpy |
| Frontend | HTML, CSS (Bootstrap 5), JavaScript |
| Charts | Chart.js |
| Database | JSON file (lightweight storage) |

---

## Project Structure

```
disease-prediction/
├── app.py                    # Main Flask application
├── train_models.py           # Script to train ML models
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── models/                  # Trained ML model files
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── breast_cancer_model.pkl
│   └── breast_cancer_scaler.pkl
├── templates/               # HTML templates
│   ├── index.html           # Home page
│   ├── diabetes.html        # Diabetes prediction form
│   ├── heart.html           # Heart disease prediction form
│   ├── breast_cancer.html   # Breast cancer prediction form
│   └── dashboard.html       # Analytics dashboard
├── static/                  # Static files
│   └── css/
│       └── style.css        # Custom styles
└── datasets/                # Training datasets (optional)
```

---

## Datasets Used

### 1. Diabetes Dataset
- **Source**: Pima Indians Diabetes Database (UCI Machine Learning Repository)
- **Samples**: 2000 synthetic samples based on original dataset statistics
- **Features**: 8 (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Target**: Binary (Diabetic / Non-Diabetic)

### 2. Heart Disease Dataset
- **Source**: UCI Heart Disease Dataset (Cleveland Clinic Foundation)
- **Samples**: 2000 synthetic samples based on original dataset statistics
- **Features**: 13 (Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Induced Angina, ST Depression, Slope, Number of Major Vessels, Thalassemia)
- **Target**: Binary (Heart Disease / No Heart Disease)

### 3. Breast Cancer Dataset
- **Source**: Wisconsin Breast Cancer Dataset (sklearn built-in)
- **Samples**: 569 (original dataset)
- **Features**: 30 (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension - Mean, SE, and Worst values)
- **Target**: Binary (Malignant / Benign)

---

## How It Works

### Data Preprocessing
1. **Data Loading**: Load datasets from CSV files or generate synthetic data
2. **Feature Scaling**: Apply StandardScaler to normalize features
3. **Train-Test Split**: Split data into 80% training and 20% testing sets

### Model Training
1. **Diabetes Model**: Random Forest Classifier with 100 estimators
2. **Heart Disease Model**: Logistic Regression with max iterations = 1000
3. **Breast Cancer Model**: Support Vector Machine (SVM) with RBF kernel

### Prediction Pipeline
1. User fills the form with health parameters
2. Input is scaled using the trained scaler
3. Model makes prediction (0 or 1)
4. Model provides probability scores
5. Result is displayed with confidence percentage
6. Prediction is saved to history

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the ML Models

```bash
python train_models.py
```

This will:
- Generate training datasets
- Train 3 ML models (Diabetes, Heart Disease, Breast Cancer)
- Save the trained models to the `models/` directory

Expected output:
```
Training Diabetes Prediction Model...
  Accuracy: 94.00%
Training Heart Disease Prediction Model...
  Accuracy: 81.50%
Training Breast Cancer Prediction Model...
  Accuracy: 98.25%
```

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Open in Browser

Visit: **http://localhost:5000**

---

## How to Use

1. **Home Page** - Select which disease you want to predict
2. **Fill the Form** - Enter the required health parameters
   - For Diabetes: Select gender first (pregnancy field appears only for females)
3. **Get Prediction** - Click the predict button to see results
4. **View Dashboard** - Check the analytics dashboard for prediction history

---

## Model Details

### Diabetes Prediction
- **Algorithm**: Random Forest Classifier
- **Features**: 8 (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Accuracy**: ~94%
- **Note**: Pregnancy field is only shown for female users

### Heart Disease Prediction
- **Algorithm**: Logistic Regression
- **Features**: 13 (Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, etc.)
- **Accuracy**: ~81.5%

### Breast Cancer Classification
- **Algorithm**: Support Vector Machine (SVM)
- **Features**: 30 (Radius, Texture, Perimeter, Area, Smoothness, etc. - Mean, SE, and Worst values)
- **Accuracy**: ~98.25%

---

## Important Notes

- This system is for **educational purposes only**
- It is **not a substitute** for professional medical advice, diagnosis, or treatment
- **Predictions may not be accurate** - Always consult an experienced doctor for proper medical diagnosis and treatment
- The models are trained on limited datasets and may not work well for all populations
- Results should be used as a reference only, not as a medical diagnosis

---

## Medical Disclaimer

**WARNING**: This application is designed for educational and academic purposes only. The predictions generated by this system are based on statistical models and machine learning algorithms trained on specific datasets. These predictions:

- May not be accurate for individual cases
- Should not be used as a basis for medical decisions
- Are not a replacement for professional medical consultation
- Should always be verified by a qualified healthcare provider

**Always consult an experienced doctor or medical professional for proper diagnosis, treatment, and medical advice.**

---

## For Developers

### Adding New Diseases

1. Create a new training script in `train_models.py`
2. Add a new route in `app.py`
3. Create a new HTML template in `templates/`
4. Update the home page to include the new disease

### Retraining Models

To retrain models with updated data:

```bash
python train_models.py
```

The new models will be saved automatically to the `models/` directory.

---

## Created By

**S AKASH DORA**

MCA Project 2026

---

## License

This project is for academic use. Feel free to modify and use for your MCA project.

---

## Acknowledgments

- Dataset sources: UCI Machine Learning Repository, Kaggle, scikit-learn
- Built as part of MCA curriculum
- Technologies: Flask, scikit-learn, Bootstrap, Chart.js

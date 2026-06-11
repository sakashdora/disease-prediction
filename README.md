# Disease Prediction System

A machine learning-based web application that predicts the risk of **Diabetes**, **Heart Disease**, and **Breast Cancer** using trained ML models. Built with Python, Flask, and scikit-learn.

---

## Features

- **Diabetes Prediction** - Predicts diabetes risk using glucose level, BMI, age, and other health parameters
- **Heart Disease Prediction** - Assesses heart disease risk using cholesterol, blood pressure, ECG results, and more
- **Breast Cancer Classification** - Classifies breast cancer as benign or malignant using cell nucleus characteristics
- **Analytics Dashboard** - Visualizes prediction history with charts and statistics
- **Prediction History** - Stores all predictions for later review

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
│   ├── css/
│   │   └── style.css        # Custom styles
│   └── js/
│       └── charts.js        # Chart configurations
└── datasets/                # Training datasets (optional)
```

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
3. **Get Prediction** - Click the predict button to see results
4. **View Dashboard** - Check the analytics dashboard for prediction history

---

## Model Details

### Diabetes Prediction
- **Algorithm**: Random Forest Classifier
- **Features**: 8 (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
- **Accuracy**: ~94%

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
- Always consult a qualified healthcare provider for medical concerns
- Predictions are based on statistical models and may not be accurate for all cases

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

## License

This project is for academic use. Feel free to modify and use for your MCA project.

---

## Acknowledgments

- Dataset sources: UCI Machine Learning Repository, Kaggle
- Built as part of MCA curriculum
- Technologies: Flask, scikit-learn, Bootstrap, Chart.js

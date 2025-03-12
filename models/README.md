# ✈️ Airfare Price Prediction Models

This directory contains trained machine learning models for predicting airfare prices on domestic flights departing from **Los Angeles International Airport (LAX)** during peak **summer months (June–August).**

## 📂 Available Models

| Model | File | Framework | Purpose |
|-------|------|-----------|---------|
| **Random Forest (Primary Model)** | `random_forest_model.pkl` | Scikit-learn | Best-performing model for fare prediction |
| **Linear Regression (Baseline Model)** | `linear_regression_model.pkl` | Scikit-learn | Simple interpretable benchmark |
| **LSTM (Sequential Model)** | `lstm_model.keras` | TensorFlow/Keras | Attempts to capture sequential fare trends |

## 🚀 How to Load the Models

### ✅ **Loading Random Forest or Linear Regression (Scikit-Learn)**
```python
import joblib

# Load trained model
model = joblib.load('models/random_forest_model.pkl')

# Generate predictions
predictions = model.predict(X_test)
```

### ✅ **Loading LSTM (TensorFlow/Keras)**
```python
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('models/lstm_model.keras')

# Generate predictions
predictions = model.predict(X_test_sequence)
```

## 📊 **Model Performance Summary**
| Model | MAE | R² Score |
|--------|------|---------|
| **Random Forest** | 0.0073 | 0.9997 |
| **Linear Regression** | 0.2250 | 0.4167 |
| **LSTM** | 120.37 | -0.0457 |

## 📌 Model Versioning
- **Current Version:** `v1.1`
- **Recent Updates:**
  - Optimized **Random Forest** hyperparameters (`n_estimators=50`, `max_depth=20`).
  - Improved **feature selection** using **SHAP analysis**.
  - LSTM trained on **2M rows** (previously full dataset).

---

📖 **For full details on model training and evaluation, see** [Model Documentation](../docs/model_documentation.md).

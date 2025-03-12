# ✈️ Airfare Price Prediction Models

This directory contains trained machine learning models for predicting airfare prices on domestic flights departing from **Los Angeles International Airport (LAX)** during peak **summer months (June–August).**

## 📂 Available Models

| Model | File | Framework | Purpose |
|-------|------|-----------|---------|
| **Random Forest (Primary Model)** | `random_forest_model.pkl` | Scikit-learn | Best-performing model for fare prediction |
| **Linear Regression (Baseline Model)** | `linear_regression_model.pkl` | Scikit-learn | Simple interpretable benchmark |
| **LSTM (Sequential Model)** | `lstm_model.keras` | TensorFlow/Keras | Attempts to capture sequential fare trends |

## 🚀 How to Load and Evaluate the Models

Each trained model is stored in the `models/` directory and can be loaded for inference.

### ✅ **Loading and Evaluating Random Forest Linear Regression (Scikit-Learn)**
1. Load the test data from `test_data/random_forest/`
2. Load the trained `random_forest.pkl` model
3. Generate predictions and compute evaluation metrics

```python
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# Load test data
X_test = joblib.load("test_data/RandomForest/X_test_rf.pkl")
y_test = joblib.load("test_data/RandomForest/y_test_rf.pkl")

# Load trained model
rf_model = joblib.load("models/random_forest.pkl")

# Make predictions
predictions = rf_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Random Forest - MAE: {mae}")
print(f"Random Forest - R² Score: {r2}")

```

## ✅ **Loading and Evaluating Linear Regression (Scikit-Learn)**
1. Load the test data from `test_data/LinearRegression/`
2. Load the trained `linear_regression.pkl` model
3. Generate predictions and compute evaluation metrics

```python
# Load test data
X_test = joblib.load("test_data/LinearRegression/X_test_lr.pkl")
y_test = joblib.load("test_data/LinearRegression/y_test_lr.pkl")

# Load trained model
lr_model = joblib.load("models/linear_regression.pkl")

# Make predictions
predictions = lr_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Linear Regression - MAE: {mae}")
print(f"Linear Regression - R² Score: {r2}")

```

### ✅ **Loading and Evaluating LSTM (TensorFlow/Keras)**
### ✅ Evaluating LSTM Model
1. Load the test data from `test_data/lstm/`
2. Load the trained `lstm_model.h5` model
3. Compute evaluation metrics

```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score

# Load test data
X_test = np.load("test_data/LSTM/X_test_lstm.npy")
y_test = np.load("test_data/LSTM/y_test_lstm.npy")

# Load trained LSTM model
lstm_model = load_model("models/lstm_model.h5")

# Make predictions
predictions = lstm_model.predict(X_test_scaled).flatten()

# Compute evaluation metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"LSTM - MAE: {mae}")
print(f"LSTM - R² Score: {r2}")

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

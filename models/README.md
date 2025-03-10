# Models Documentation

This directory contains trained and serialized machine learning models used for predicting airfare prices for domestic flights departing from Los Angeles (LAX) during peak summer months (June–August). Each model was developed, trained, evaluated, and version-controlled to ensure reproducibility and efficient model management.

---

## Included Models

### ✅ 1. **Random Forest Regression (Primary Model)**
- **File:** `random_forest_model.pkl`
- **Framework:** scikit-learn
- **Purpose:** Primary prediction model; chosen for its superior accuracy, handling of nonlinear data, and robustness to outliers.
- **Performance:**
  - **Mean Absolute Error (MAE):** 
  - **R² Score:** 
- **Features used:** 
  - 

---

### ✅ 2. **Linear Regression (Baseline Model)**
- **Filename:** `linear_regression_model.pkl`
- **Framework:** scikit-learn
- **Purpose:** Serves as an interpretable baseline comparison.
- **Performance:**
  - **Mean Absolute Error (MAE):** 
  - **R² Score:** 
- **Limitations:** Struggled with nonlinearities in pricing data; high sensitivity to outliers.

---

### ✅ 3. **LSTM (Long Short-Term Memory Neural Network)**
- **Filename:** `lstm_model.keras`
- **Framework:** TensorFlow/Keras
- **Purpose:** Designed to capture short-term sequential pricing trends.
- **Performance:**
  - **Mean Absolute Error (MAE):** 
  - **R² Score:** 
- **Limitations:** Limited historical data restricted its performance, indicating that additional data may enhance future predictions.

---

## 🚀 How to Load the Models

### ✅ **Loading Random Forest or Linear Regression (scikit-learn):**
```python
import joblib

# Load trained Random Forest or Linear Regression model
model = joblib.load('models/random_forest_model.pkl')

# Generate predictions
predictions = model.predict(X_test) 

```
### ✅ **Loading LSTM (TensorFlow/Keras):**
```python
from tensorflow.keras.models import load_model

# Load trained LSTM model
model = load_model('models/lstm_model.keras')

# Generate predictions
predictions = model.predict(X_test_sequence)
```
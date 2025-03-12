# Models Documentation

This directory contains trained and serialized machine learning models used for predicting airfare prices for domestic flights departing from Los Angeles (LAX) during peak summer months (June–August). Each model was developed, trained, evaluated, and version-controlled to ensure reproducibility and efficient model management.

---
## 📊 Dataset Used

- **Source:** Historical airfare pricing data from Kaggle.
- **Size:** XX,XXX rows and XX columns.
- **Key Features:**
  - `pricePerMile`: Cost per mile traveled.
  - `fareLag_1`: Previous day's airfare price.
  - `totalTravelDistance`: Total flight distance.
- **Preprocessing:**
  - **Log transformation** applied to `totalFare` to reduce skewness.
  - **Categorical features** encoded using frequency encoding.
  - **Highly correlated features** removed (correlation > 0.85).

---

## Included Models

The repository includes three models, each serving a distinct purpose:

### ✅ 1. **Random Forest Regression (Primary Model)**
- **File:** `random_forest_model.pkl`
- **Framework:** scikit-learn
- **Purpose:** 
    - Selected for its high accuracy, ability to handle nonlinear relationships, and robustness to outliers.
    - Used as the primary prediction model for airfare prices.
 - **Structure & Parameters:**
  - **Number of Trees (`n_estimators`)**: 100
  - **Max Depth (`max_depth`)**: 20
  - **Min Samples Split (`min_samples_split`)**: 5
  - **Min Samples Leaf (`min_samples_leaf`)**: 2
  - **Feature Selection:** Used **permutation importance** and **SHAP analysis**.
#### 📊 **Performance**
  - **Mean Absolute Error (MAE):** `0.0110`
  - **R² Score:** `0.9997`
**- **Features used:** **
  - Feature selection based on permutation importance and SHAP analysis.
  - Below are the top features used in the Random Forest Regression model, ranked by importance:

    | Feature              | Importance Score |
    |----------------------|----------------:|
    | pricePerMile        | 0.444068         |
    | fareLag_1           | 0.242052         |
    | totalTravelDistance | 0.092293         |

  > **Interpretation:** The **higher the score**, the greater the feature's impact on predicting airfare prices.
#### ⚠️ **Constraints & Limitations**
- Performs well **when past airfare trends are stable**.
- **Struggles with sudden, extreme airfare fluctuations** (e.g., flash sales, emergency price hikes).
- May not **generalize well** to airlines/routes not present in the training data.


---

### ✅ 2. **Linear Regression (Baseline Model)**
- **Filename:** `linear_regression_model.pkl`
- **Framework:** scikit-learn
- **Purpose:**
   - Serves as a simple and interpretable baseline comparison.
    - Helps assess whether a simpler model can perform well against more complex models.
 - **Structure & Parameters:**  
    - Standard Ordinary Least Squares (OLS) regression.
#### 📊 **Performance**
  - **Mean Absolute Error (MAE):** `0.2250`
  - **R² Score:** `0.4167`
#### ⚠️ **Constraints & Limitations**
- **Struggles with nonlinearities** in airfare pricing data.  
- **Highly sensitive to outliers**, leading to reduced predictive power.  
- **Performed significantly worse than Random Forest**, indicating that airfare pricing patterns are not well captured by a simple linear model.
---

### ✅ 3. **LSTM (Long Short-Term Memory Neural Network)**
- **Filename:** `lstm_model.keras`
- **Framework:** TensorFlow/Keras
- **Purpose:**
   - Designed to capture short-term sequential pricing trends.
   - Evaluates how past airfare fluctuations affect future prices.
- **Performance:**
  - **Mean Absolute Error (MAE):** `120.37`
  - **R² Score:** `-.0457`
- **Limitations:**
    - Limited historical data restricted its ability to learn meaningful trends.  
    - LSTMs typically require large amounts of sequential data to perform well.  
    - Additional data and feature engineering could significantly improve model performance.
.
---

## 🔄 Model Versioning
- **Current Version:** `v1.0`
- **Changes from Previous Versions:**
  - **Feature selection refined** (Permutation importance + SHAP added).
  - **Hyperparameter tuning improved** for Random Forest.
  - **More historical data added** for LSTM.
- **Planned Future Updates:**
  - Expand **LSTM training dataset**.
  - Experiment with **XGBoost and Gradient Boosting**.

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

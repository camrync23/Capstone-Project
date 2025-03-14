# Models Documentation

This directory contains trained and serialized machine learning models used for predicting airfare prices for domestic flights departing from Los Angeles (LAX) during peak summer months (June–August). Each model was developed, trained, evaluated, and version-controlled to ensure reproducibility and efficient model management.

---
## 📊 Dataset Used

- **Source:** [Expedia Flight Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/dilwong/flightprices)  
- **Scope:** Flights departing from **LAX (Los Angeles International Airport)** during peak **summer months (June – August)**  
- **Size:** **~3.8 million instances** (filtered from ~80 million records)  

For full details on dataset collection, features, and preprocessing steps, see the **[Dataset Documentation](../docs/dataset_documentation.md).**

### 🔹 **Key Features for Model Training**
- `pricePerMile` – Normalized cost per mile traveled.  
- `fareLag_1` – Airfare price from **one day prior** (captures short-term fluctuations).  
- `totalTravelDistance` – Total flight distance in miles.  
- `daysToDeparture` – Days between booking and flight date.  
- `totalLayoverTime` – Sum of layover durations (in minutes).  
- `isHoliday` – Indicator for flights on/near major U.S. holidays.  

### 🛠️ **Preprocessing (Summary)**
- **Log transformation** applied to `totalFare` to handle skewness.  
- **Categorical features** encoded (one-hot for nominal, ordinal for ranked).  
- **Feature scaling:** Min-Max Scaling for numerical variables.  
- **Feature selection:** Based on **SHAP Analysis** & **Permutation Importance**.  

👉 **For a full breakdown of data cleaning, feature engineering, and outlier handling, refer to** [Dataset Documentation](../docs/dataset_documentation.md).

---

## 📌 Included Models

The repository includes three models, each serving a distinct purpose:

### ✅ 1. **Random Forest Regression (Primary Model)**
- **File Download** `random_forest_download.py` contains the instructions to download the random forest model. Model was too large to store in GitHub
- **File:** `random_forest.pkl`
- **Framework:** scikit-learn
- **Purpose:** 
    - Selected for its high accuracy, ability to handle nonlinear relationships, and robustness to outliers.
    - Serves as the primary prediction model for airfare prices.
 - **Structure & Parameters:**
  - **Number of Trees (`n_estimators`)**: 50
  - **Max Depth (`max_depth`)**: 20
  - **Min Samples Split (`min_samples_split`)**: 10
  - **Min Samples Leaf (`min_samples_leaf`)**: 2
  - **Feature Selection:** Used **permutation importance** and **SHAP analysis**.
#### 📊 **Performance**
  - **Root Mean Squared Error (RMSE) :** '`0.0081`
  - **Mean Absolute Error (MAE):** `0.0073`
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
- **Filename:** `linear_regression.pkl`
- **Framework:** scikit-learn
- **Purpose:**
   - Serves as a simple and interpretable baseline comparison.
    - Helps assess whether a simpler model can perform well against more complex models.
#### 🔹 **Model Structure & Variants**
- **Ordinary Least Squares (OLS) Regression** (Baseline model)
- **Ridge Regression** (Handles multicollinearity, optimized with `alpha=10`)
- **Lasso Regression** (Feature selection, `alpha=0.01`)
- **Polynomial Regression (Degree=2)** (Captures some nonlinear relationships)
- **Feature Selection with Recursive Feature Elimination (RFE)**  
  - **Top 5 Selected Features:**
    - `pricePerMile`
    - `postHolidayFlight`
    - `totalLayoverTime`
    - `durationToDistanceRatio`
    - `cabin_classes_ordinal`

#### 📊 **Performance Comparison**
| Model Type                       | Mean Absolute Error (MAE) | R² Score |
|----------------------------------|-------------------------|---------:|
| **Ordinary Least Squares (OLS)** | `0.2250`                | `0.4167` |
| **Ridge Regression (`alpha=10`)** | `0.2250`                | `0.4166` |
| **Lasso Regression (`alpha=0.01`)** | `0.2927`                | `0.1158` |
| **Polynomial Regression (Degree=2)** | `0.2169`                | `0.4801` |
| **Ridge (Top 5 Selected Features)** | `0.2254`                | `0.4149` |

#### ⚠️ **Constraints & Limitations**
- **Struggles with nonlinearities** in airfare pricing data.  
- **Highly sensitive to outliers**, which **Ridge/Lasso partially address**.  
- **Polynomial Regression improves performance** but **adds complexity**.  
- **RFE shows that only 5 key features contribute meaningfully**, which may help simplify the model.

---

### ✅ 3. **LSTM (Long Short-Term Memory Neural Network)**
- **Filename:** `lstm_model.h5`
- **Framework:** TensorFlow
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
## 🔍 Feature & Hyperparameter Tuning

| Model              | Tuning Method |
|--------------------|--------------|
| **Random Forest**  | Used `RandomizedSearchCV` to optimize `n_estimators`, `max_depth`, `min_samples_split` |
| **Linear Regression** | Used `GridSearchCV` to optimize `alpha` for Ridge & Lasso |
| **LSTM**          | Tuned `LSTM units`, `dropout rate`, and `batch size` through manual hyperparameter search |

---

## 🔄 Model Evaluation & Testing

| Model                            | Train-Test Split                 | Evaluation Metrics |
|----------------------------------|---------------------------------|--------------------|
| **Random Forest & Linear Regression** | **80% training, 10% validation, 10% test** | **RMSE, MAE, R² Score** |
| **LSTM**                         | **Time-series split (80%-10%-10%)** | **MAE, R² Score** |

---

## 🔄 Model Versioning

### **Current Version: `v1.1`**

#### 🔹 **Changes from Previous Versions**
- **Random Forest**
  - Updated `n_estimators=50` (**previously 100**) for better performance.
  - Refined feature selection using **SHAP analysis**.
- **Linear Regression**
  - Optimized **Ridge & Lasso** hyperparameters using **GridSearchCV**.
- **LSTM**
  - Trained on **2M rows** (**previously full dataset**).
  - Further optimized **model architecture**.

#### 🔹 **Planned Future Updates**
- Expand **LSTM training dataset** for better sequential learning.
- Experiment with **XGBoost & Gradient Boosting** as alternatives to **Random Forest**.
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

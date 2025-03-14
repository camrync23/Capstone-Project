# ✈️ Capstone Project: Dynamic Flight Price Prediction & Fare Optimization

## 📌 Overview
This project aims to predict airfare prices for flights departing from **Los Angeles International Airport (LAX)** during peak **summer months (June–August)**. The goal is to develop a machine learning model that helps passengers identify the best time to book flights, using **historical pricing data and key influencing factors** such as seasonality, airline, route, demand, and time-to-departure.

## 📂 Repository Structure
```
Capstone-Project/
│── .dvc/                         # DVC tracking for data & models
│── config/                       # Configurations for pipeline & models
│   ├── main.yaml                 # Main configuration file
│── data/                         # Data storage and preprocessing
│   ├── 01_raw/                   # Original raw dataset
│   ├── 02_filtered/              # Filtered dataset (LAX, summer months)
│   ├── 03_processed/             # Final processed dataset (ready for training)
│── test_data/                    # Dedicated folder for test data (ensuring reproducibility)
│   ├── RandomForest/             # Test data specifically for the Random Forest model
│   │   ├── X_test_rf.pkl         # Feature test set for Random Forest
│   │   ├── y_test_rf.pkl         # Target variable test set for Random Forest
│   ├── LinearRegression/         # Test data specifically for the Linear Regression model
│   │   ├── X_test_lr.pkl         # Feature test set for Linear Regression
│   │   ├── y_test_lr.pkl         # Target variable test set for Linear Regression
│   ├── LSTM/                     # Test data specifically for the LSTM model
│   │   ├── X_test_lstm.npy       # Feature test set for LSTM (NumPy format for 3D tensors)
│   │   ├── y_test_lstm.npy       # Target variable test set for LSTM
│── docs/                         # Project documentation
│   ├── dataset_documentation.md  # Dataset details
│   ├── model_documentation.md    # Model architecture & training details
│   ├── system_documentation.md   # System & pipeline documentation
│── models/                       # Saved trained models for reproducibility
│   ├── EvaluateModels.ipynb      # Notebook that can be used to reproduce results from trained models 
│   ├── linear_regression.pkl     # Linear Regression model
│   ├── random_forest_download.py # Python file used to download trained random forest model 
│   ├── lstm_model.h5             # LSTM model (Keras/TensorFlow format)
│   ├── README.md                 # README provided detailing how to use the models and evaluation notebook
│── notebooks/                    # Jupyter Notebooks for exploratory analysis and experiments
│   ├── DataExploration.ipynb     # Initial exploratory analysis
│   ├── DataPreprocessing.ipynb   # Data cleaning & transformation
│   ├── LSTMv1.ipynb              # LSTM model training and tuning version 1
│   ├── LSTMv2.ipynb              # LSTM model training and tuning version 2
│   ├── LinearRegression.ipynb    # Linear regression experiments
│   ├── RandomForestv1.ipynb      # Random forest training & hyperparameter tuning version 1
│   ├── RandomForestv2.ipynb      # Random forest training & hyperparameter tuning version 2
│── .dvcignore                    # Ignore files for DVC tracking
│── .gitignore                    # Ignore files for Git tracking
│── README.md                     # Project overview & instructions
│── requirements.txt              # Python dependencies
```

## 🚀 How to Run the Project

### 1️⃣ **Set Up the Environment**
Ensure all dependencies are installed:
```sh
pip install -r requirements.txt
```

### 2️⃣ **Download Necessary Files**
Since the Random Forest model is too large for GitHub, it is stored in Google Drive. Run the following script to download and extract the model:
```sh
python models/random_forest_download.py
```

### 3️⃣ **Run the Model Evaluation**
Use the provided notebook to evaluate the models:
Open the Jupyter Notebook:
```sh
jupyter notebook notebooks/EvaluateModels.ipynb
```

## 📊 Available Models

| Model                        | File                      | Framework          | Purpose                                   |
|------------------------------|---------------------------|--------------------|-------------------------------------------|
| Random Forest (Primary Model) | random_forest.pkl (downloaded) | Scikit-learn    | Best-performing model for fare prediction |
| Linear Regression (Baseline Model) | linear_regression.pkl | Scikit-learn     | Simple interpretable benchmark           |
| LSTM (Sequential Model)      | lstm_model.h5            | TensorFlow/Keras  | Attempts to capture sequential fare trends |

## 📊 Model Performance Summary

| Model            | RMSE   | MAE    | R² Score |
|-----------------|--------|--------|----------|
| Random Forest   | 0.0081 | 0.0110 | 0.9997   |
| Linear Regression | N/A    | 0.2250 | 0.4167   |
| LSTM            | N/A    | 120.37 | -0.0457  |


## 📖 Documentation
- **Dataset Documentation**
- **Model Training Details**
- **System Pipeline Overview**

## 📌 Authors
- **Allison Conrey** - [alconrey@ucsd.edu](mailto:alconrey@ucsd.edu)
- **Camryn Curtis** - [ccurtis@ucsd.edu](mailto:ccurtis@ucsd.edu)

## 📌 Project Abstract

### Background
Flight pricing is dynamic and influenced by factors such as demand, seasonality, distance, and availability. Price variability is particularly pronounced during peak travel seasons, such as summer. Our goal is to develop a predictive model to estimate ticket prices based on key variables, including travel dates, distance, airline, and ticket characteristics.

### Problem Definition
Our project addresses two key questions:
1. Can we predict airfare prices to assist travelers in knowing the best time to book?
2. What features contribute to airfare pricing?

Using the Expedia dataset, our model will incorporate features such as:
- Departure and arrival dates
- Travel distance
- Cabin class
- Airline
- Ticket attributes (e.g., refundable, non-stop, etc.)

The output of our model will be the predicted ticket price.

### Success Criteria
Our model's success will be measured using:
- **Mean Absolute Error (MAE)** for prediction accuracy
- **R-Squared (R²)** to measure variance explanation
- **Feature Importance Analysis** to identify key drivers of price
- **Comparison between Linear Regression, LSTM, and Random Forest** to assess trade-offs between interpretability and performance

## 📜 License
This project is for educational purposes and follows an open-source approach.


## References
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
- Chavan, A., Rathod, I., & Bobde, S. (2024). Comparative analysis of machine learning models for accurate flight price prediction. *International Journal of Innovative Science and Research Technology (IJISRT), 9(9),* 2798–2805. [DOI](https://doi.org/10.38124/ijisrt/IJISRT24SEP1688)
- Etzioni, O., Tuchinda, R., Knoblock, C. A., & Yates, A. (2003). To buy or not to buy: Mining airfare data to minimize ticket purchase price. *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
- Guan, X. (2024). Flight price prediction web-based platform: Leveraging generative AI for real-time airfare forecasting. *School of Tourism, Hainan University.* Received December 9, 2023; Accepted January 8, 2024; Published April 8, 2024.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. [https://doi.org/10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
- Krishna, N. G., Iswarya, G., Narasimharao, B. S., Durga Devi, B. N., & Dani, B. (2024). Predicting airline ticket prices using machine learning. *International Journal of Scientific Research in Engineering and Management (IJSREM), 8(4),* 1–5. [DOI](https://doi.org/10.55041/IJSREM31185)
- Molnar, C. (2020). *Interpretable machine learning: A guide for making black box models explainable*. Springer. [https://doi.org/10.1007/978-3-030-30311-8](https://doi.org/10.1007/978-3-030-30311-8)
- Sharma, N., Singh, M., & Jain, R. (2024). Evaluating the effectiveness of LSTM networks for airfare prediction: A time-series analysis. *Journal of Machine Learning for Economics, 22*(1), 75-89.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B, 58*(1), 267-288.
- Wang, W., Chen, P., Huang, Y., & Wang, X. (2019). A framework for airfare price prediction: A machine learning approach. *Proceedings of the 2019 IEEE International Conference on Information Reuse and Integration (IRI)*, 127-134. UMKC Library.


---
This README serves as an overview of our project and will be updated with additional details as our research progresses.


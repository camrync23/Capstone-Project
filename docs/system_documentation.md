# 🏗️ System Documentation

## 1. System Overview
This system is designed to **predict airfare prices** (specifically for flights departing from LAX) using a structured data pipeline and multiple machine learning models. It was developed as part of the **288R, WI 2024** course under Prof. Umesh Bellur.

### Purpose
1. Enable **data-driven insights** for travelers who want to minimize airfare costs.
2. Provide a **scalable, maintainable** pipeline that can be updated as data changes.

### Scope
- **Data Pipeline** that ingests, cleans, and transforms flight data.
- **Model Training & Evaluation** for airfare price prediction (Random Forest, LSTM, Linear Regression, etc.).
- **Deployment & Maintenance** guidelines for continuous improvement and updating the system.

---

## 2. System Workflow

Below is a high-level overview of the **end-to-end workflow**, from raw data to producing flight-price predictions.

    Raw Expedia Data (Snappy Parquet)
           ↓
     [Data Ingestion]
           ↓
     [Data Preprocessing & Feature Engineering]
           ↓
     [Model Training (RF, LSTM, LinReg)]
           ↓
     [Model Evaluation & Selection]
           ↓
     [Predictions & Recommendations]

1. **Data Ingestion**  
   - Loads the Expedia dataset (Snappy Parquet format from DropBox provided on Kaggle).  
   - Filters flights to only those departing from **LAX**.  
   - Export as **Parquet** (Snappy compressed) for efficient storage.  

2. **Data Preprocessing**  
   - **Cleaning**: Handles missing values (e.g., removes rows with missing `flightDistance`), checks for duplicates.  
   - **Encoding**: One-hot encoding for nominal categorical variables (e.g., airline, destinationAirport).  Ordinal encoding for categorical variables with inherit ordering (e.g., cabin class)
   - **Scaling**: Min-Max scaling for numeric features (fare, distance, etc.  
   - **Feature Engineering**: Creates derived features like `daysToDeparture`, `pricePerMile`, and holiday indicators (`isHoliday`).

3. **Model Training**  
   - Trains **multiple ML models**:  
       - **Random Forest (Primary)**  
       - **LSTM Neural Network** (leveraging sequential patterns if enough time-series data is available)  
       - **Linear Regression** (baseline)  
   - Hyperparameter tuning through **Grid Search** or **Random Search**.

4. **Model Evaluation & Selection**  
   - Uses **Mean Absolute Error (MAE)** and **R²** as primary metrics.  
   - Compares performance of all trained models.  
   - In prior runs, **Random Forest** demonstrated the best accuracy with an MAE of around **\$11**.

5. **Predictions & Recommendations**  
   - The best-performing model (e.g., Random Forest) generates flight-price predictions.  
   - Results inform users about **optimal booking windows** (when to buy tickets).

---

## 3. Tools & Must-Have Components

To successfully operate the system, the following tools and services **must** exist:

1. **Data Storage**  
   - Local file system or cloud storage (e.g., AWS S3, Azure Blob, or Google Cloud Storage) to hold raw and cleaned flight data.

2. **Computational Environment**  
   - Python 3.9+ interpreter (preferably in a virtualenv or Conda environment).  
   - Jupyter Notebooks or VS Code for data exploration and iterative model development.  
   - GPU/TPU (optional) for LSTM training if performance is critical.

3. **Python Libraries**

   | Dependency      | Version  | Purpose                       |
   |-----------------|----------|-------------------------------|
   | pandas          | >=1.3    | Data manipulation             |
   | numpy           | >=1.21   | Numerical computations        |
   | scikit-learn    | >=1.0    | Classical ML models           |
   | tensorflow      | >=2.6    | Deep learning (LSTM)          |
   | matplotlib      | >=3.4    | Visualization                 |
   | seaborn         | >=0.11   | Statistical graphics          |
   | dvc            | >=2.8    | Data & model versioning (opt) |

4. **Collaboration & Version Control**  
   - **Git** for source code versioning.  
   - **GitHub / GitLab** repository for team collaboration.  
   - **DVC** (Data Version Control) to track large dataset versions (optional but recommended).

5. **Holiday / Calendar Service** (optional)  
   - If holiday or seasonal indicators are needed, an external holiday API or offline calendar data might be integrated into the feature-engineering step.

---

## 4. How Updates to the System Are Performed

### 4.1 Data Updates
- **Frequency**: If new Expedia data is released or additional months are appended, the ingestion script is rerun to filter flights from LAX and apply the same cleaning transformations.  
- **Data Validation**: Includes checks for missing essential columns (e.g., `flightDistance`, `totalFare`), verifying data schema, and scanning for duplicates.

### 4.2 Model Updates
- **Retraining Trigger**:  
   1. New data is ingested or existing data is significantly updated.  
   2. Performance drops are observed in real usage.  
   3. Modifications in feature engineering or hyperparameters.  
- **Retraining Steps**:  
   1. Run data preprocessing scripts (e.g., `src/preprocess.py`).  
   2. Retrain model(s) (e.g., `src/train_model.py`).  
   3. Evaluate performance (e.g., `src/evaluate_model.py`).  
- **Versioning**:  
   - Tag each new model artifact (e.g., `random_forest_v2.pkl`).  
   - Commit changes to Git and push updates.  
   - DVC can track the new model file and data changes.

### 4.3 Deployment Updates (Optional)
- **If Deployed as an API**:  
   1. Build a Docker image with updated code and model.  
   2. Push to a container registry (e.g., Docker Hub).  
   3. Use a CI/CD pipeline (GitHub Actions/Jenkins) for continuous delivery.
- **Rolling Updates**: The previous model stays live until the new version is validated in staging or canary environment.

---

## 5. Dependencies (Software & People)

### 5.1 Software Dependencies
- **OS**: Linux/Unix recommended (though Windows is also possible).  
- **Python Environment**: Must have all libraries from `requirements.txt` installed.
- **DVC**: If data versioning is critical, DVC must be set up properly.  
- **Holiday API**: If using holiday augmentation, the system relies on either an external API or local CSV for holiday dates.

### 5.2 People Dependencies
- **Data Scientist / Engineer**:  
   - Maintains ingestion, preprocessing scripts.  
   - Oversees model research, training, hyperparameter tuning.
- **ML Engineer / DevOps**:  
   - Manages model deployment pipelines, containerization, production environment stability.  
   - Ensures GPU resources (if needed) are available.
- **Domain Expert (Optional)**:  
   - Provides domain knowledge on airline industry or dynamic pricing.  
   - Validates that anomalies or outliers align with real-world behaviors (e.g., surge pricing, seat availability constraints).

---

## 6. Data Flow & Storage

Below is an example file structure and an explanation of data flow:

    Capstone-Project/
    ├── data/
    │   ├── raw/                (Original CSV/Parquet from Kaggle)
    │   ├── processed/          (Cleaned, feature-engineered data)
    │   └── logs/               (Logs of data ingestion, cleaning steps)
    ├── models/                 (Trained model artifacts)
    ├── src/                    (Source code: ingestion, preprocessing, training)
    │   ├── ingest_data.py
    │   ├── preprocess.py
    │   ├── train_model.py
    │   ├── evaluate_model.py
    │   └── utils/              (Helper scripts)
    ├── experiments/            (Jupyter notebooks for EDA, preprocessing, model building)
    │   ├── data_exploration.ipynb   
    │   ├── dataset_preprocessing.ipynb
    │   ├── RandomForest.ipynb
    │   ├── LSTM.ipynb
    │   └── LinearRegression.ipynb  
    ├── docs/                   (Documentation including this file)
    │   ├── dataset_documentation.md    
    │   ├── model_documentation.md 
    │   └── system_documentation.md  
    ├── requirements.txt
    ├── .gitignore
    └── .dvc/                   (If using DVC for data and model versioning)

**Data Flow**:
1. **Raw Data** is placed in `data/raw/`.  
2. **Data Cleaning & Preprocessing** outputs to `data/processed/`.  
3. **Model Training** uses `data/processed/` and saves models in `models/`.  
4. **Logs & Documentation** track each step for reproducibility.

---

## 7. Other Models Within the System

Although Random Forest is our primary production model, the system can house multiple models:

1. **Random Forest Regressor**
   - Typically stored in `models/random_forest.pkl`.
   - Most robust to outliers, best overall MAE.

2. **LSTM Neural Network**
   - Exploits sequential nature of booking data if daily or time-series data is properly structured.
   - Stored as an `.h5` (Keras/TensorFlow) or PyTorch `.pt` file.

3. **Linear Regression**
   - A simple, interpretable baseline.
   - Helps confirm the *value-add* of advanced models.
   - May be stored as `models/linear_reg.pkl`.

### Model Documentation
Each model folder or artifact should contain:
- **Version** (e.g., `v1.3`)
- **Hyperparameters** (e.g., `n_estimators=100` for Random Forest)
- **Performance Metrics** (MAE, R²)
- **Last Updated** date

---

## 8. Versioning & Updates

1. **Git Flow**
   - `main` branch for stable production-ready code.
   - `dev` or feature branches for new experiments.
   - Pull Requests ensure code reviews and integration.

2. **DVC Workflow (Optional)**
   - `dvc add data/raw/flight_data.csv` to track raw dataset versions.
   - `dvc add models/random_forest.pkl` to track new or updated model weights.

3. **Continuous Integration (CI)**
   - Linting and unit tests run on every pull request.
   - Automated training scripts can run nightly or weekly.

---

## 9. System Limitations & Future Work

### 9.1 Known Limitations
1. **Single-Hub Focus**: Currently restricted to LAX departures; may not generalize to other airports without retraining.
2. **Short Historical Window**: Only a few months of data limits the LSTM’s ability to capture long-term seasonality.
3. **No Real-Time Dynamic Pricing**: The system uses daily or monthly snapshots, not real-time fare updates.

### 9.2 Future Improvements
1. **Expand Dataset**: Incorporate multiple airports and multi-year data for better coverage and seasonality patterns.
2. **Model Experimentation**: Try XGBoost or LightGBM for potentially higher accuracy and improved training speed.
3. **Deployment**: Expose predictions via a REST API or web application for external access.
4. **Explainability**: Add **SHAP** or **LIME** analyses to provide interpretability into which features drive flight prices.

---

## 10. Conclusion

This documentation outlines how the **flight-price–prediction** system is designed, the essential tools and components it depends on, how it handles data, the models it employs, and the processes by which updates and maintenance occur. The system combines **classical machine learning (RF, LinReg)** and **deep learning (LSTM)** approaches to produce accurate and timely flight price predictions, guided by a robust data pipeline and version-controlled environment.

By following this structure:
- **Team members** can easily collaborate and extend or update the system.
- **Stakeholders** gain confidence in a well-defined, transparent ML process.
- **New contributors** can quickly onboard to maintain or improve the flight price prediction models.

For more details:
- Refer to `README.md` in the project root for a quickstart guide.
- See the `requirements.txt` file for Python dependencies.
- Explore `experiments/` for EDA and prototype development examples.

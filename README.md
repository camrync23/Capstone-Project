# Group Project 3: Predictive Modeling for Flight Price on Domestic Travel

## Authors
**Allison Conrey** - alconrey@ucsd.edu  
**Camryn Curtis** - ccurtis@ucsd.edu  

## Abstract

### Background
Flight pricing is dynamic and influenced by factors such as demand, seasonality, distance, and availability. Price variability is particularly pronounced during peak travel seasons, such as summer. Our goal is to develop a predictive model to estimate ticket prices based on key variables, including travel dates, distance, airline, and ticket characteristics. This model will empower travelers to make cost-effective travel decisions and help airlines optimize revenue strategies.

### Problem Definition
Our project addresses two key questions:
1. **When should a traveler purchase airfare from major US cities for summer travel?**
2. **How much can they expect to pay for that airfare?**

Using the Expedia dataset, our model will incorporate features such as:
- Departure and arrival dates
- Travel distance
- Cabin class
- Airline
- Ticket attributes (e.g., refundable, non-stop, etc.)

The output of our model will be the predicted ticket price.

### Motivation
The summer travel season is marked by high demand and fluctuating prices, making it an ideal case for machine learning-based prediction models. The availability of large historical datasets allows for effective modeling of airfare pricing, leveraging various factors such as booking time, seasonality, and demand.

Our dataset consists of an Expedia dataset (~31GB) with 80 million instances. To ensure computational feasibility while maintaining analytical integrity, we will focus on outbound flights from LAX, reducing the dataset to approximately **4 million instances**. This subset is sufficiently large to capture meaningful pricing patterns and interactions while being manageable for machine learning model training.

### Dataset and Model Approach
We will split our dataset into:
- **Training Set (80%)** - For model learning
- **Validation Set (10%)** - For hyperparameter tuning
- **Test Set (10%)** - For final evaluation

To prevent overfitting and enhance generalizability, we will apply cross-validation. Given the complexity of airfare pricing, we will explore both simple and complex modeling approaches:
- **Baseline Model:** Linear Regression for interpretability
- **Primary Model:** Random Forest for its robustness in handling non-linear relationships and feature interactions
- **Feature Engineering:** Including travel distance, departure/arrival times, seasonal indicators, and ticket attributes

### Literature Review
Several studies have explored machine learning approaches for airfare prediction:
- **Chavan, Rathod, & Bobde (2024):** Evaluated Random Forest and SVM for flight price prediction, emphasizing performance metrics.
- **Krishna et al. (2024):** Highlighted the importance of feature engineering in improving model accuracy.
- **Guan (2024):** Explored generative AI for real-time airfare forecasting.

Building on these studies, we aim to apply similar techniques, incorporating historical fare trends, feature engineering, and model comparisons to assess the trade-offs between complexity and interpretability.

### Success Criteria
Our model's success will be measured using:
- **Mean Absolute Error (MAE):** Evaluates prediction accuracy by measuring the average difference between predicted and actual ticket prices.
- **R-Squared (R²):** Measures how well the model explains ticket price variability.
- **Feature Importance Analysis:** Identifies key factors influencing airfare prices.
- **Model Comparison:** Evaluating whether the complexity of Random Forest provides significant advantages over Linear Regression.

By achieving a balance between accuracy, interpretability, and generalizability, our model will offer insights that benefit both travelers and airlines in optimizing ticket pricing strategies.

## References
1. **Chavan, A., Rathod, I., & Bobde, S. (2024).** Comparative analysis of machine learning models for accurate flight price prediction. *International Journal of Innovative Science and Research Technology (IJISRT), 9(9),* 2798–2805. [DOI](https://doi.org/10.38124/ijisrt/IJISRT24SEP1688)
2. **Krishna, N. G., Iswarya, G., Narasimharao, B. S., Durga Devi, B. N., & Dani, B. (2024).** Predicting airline ticket prices using machine learning. *International Journal of Scientific Research in Engineering and Management (IJSREM), 8(4),* 1–5. [DOI](https://doi.org/10.55041/IJSREM31185)
3. **Guan, Y. (2024).** Flight price prediction web-based platform: Leveraging generative AI for real-time airfare forecasting. *School of Tourism, Hainan University.* Received December 9, 2023; Accepted January 8, 2024; Published April 8, 2024.

---
This README serves as an overview of our project and will be updated with additional details as our research progresses.


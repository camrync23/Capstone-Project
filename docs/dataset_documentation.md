# 📊 Dataset Documentation

## **Dataset Overview**
- **Dataset Name**: Expedia Flight Prices  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/dilwong/flightprices)  
- **GitHub Mirror**: [FlightPrices Repository](https://github.com/dilwong/FlightPrices)  
- **Date Collected**: Between **April 16, 2022 – October 5, 2022**  
- **Size**: 80M records (original), reduced to ~4M (LAX flights)  
- **Geographical Scope**: Major U.S. Airports  
  - Included airports: ATL, DFW, DEN, ORD, LAX, CLT, MIA, JFK, EWR, SFO, DTW, BOS, PHL, LGA, IAD, OAK  

---

## **Data Format**
- **Original Format**: CSV (from Kaggle)  
- **Processed Format**: **Apache Parquet (Snappy Compressed)**  
- **Reason for Conversion**:
  - **Faster read/write speeds** compared to CSV.
  - **Efficient storage**: Parquet is columnar, reducing file size.
  - **Better compatibility** for handling large datasets.
- **File Storage**: The processed dataset is stored in `data/processed/` as `.parquet` files.

---

## **Column Definitions**
| Column Name | Type | Description |
|-------------|------|-------------|
| `legId` | String | Unique identifier for each flight leg |
| `searchDate` | Date | The date the ticket was found on Expedia |
| `flightDate` | Date | The actual flight date |
| `startingAirport` | Categorical | IATA code for the departure airport |
| `destinationAirport` | Categorical | IATA code for the arrival airport |
| `fareBasisCode` | String | Fare basis code (varies by airline) |
| `travelDuration` | String | Total travel time (hours:minutes) |
| `elapsedDays` | Integer | Number of elapsed days (usually 0) |
| `isBasicEconomy` | Boolean | Whether the ticket is for basic economy (0/1) |
| `isRefundable` | Boolean | Whether the ticket is refundable (0/1) |
| `isNonStop` | Boolean | Whether the flight is non-stop (0/1) |
| `baseFare` | Float | Price of the ticket in USD |
| `totalFare` | Float | Price of the ticket including taxes and fees in USD |
| `seatsRemaining` | Integer | Number of seats still available at booking time |
| `totalTravelDistance` | Float | Total travel distance (miles) |
| `segmentsDepartureTimeEpochSeconds` | String | Departure time for each leg (Unix timestamp, separated by `||`) |
| `segmentsArrivalTimeEpochSeconds` | String | Arrival time for each leg (Unix timestamp, separated by `||`) |
| `segmentsDepartureAirportCode` | String | Departure airport for each leg (`||`-separated) |
| `segmentsArrivalAirportCode` | String | Arrival airport for each leg (`||`-separated) |
| `segmentsAirlineName` | String | Airline for each leg (`||`-separated) |
| `segmentsCabinCode` | String | Cabin type (e.g., "coach") |

---


## **Data Cleaning**
- **Missing Values**:
  - The dataset contained missing values in `totalTravelDistance`, which were removed due to its importance in price prediction.
- **Duplicate Handling**:
  - No duplicate entries were found, but a policy was established to eliminate duplicates in future iterations.
- **Data Type Conversions**:
  - Boolean variables (`isBasicEconomy`, `isRefundable`, `isNonStop`) were converted to binary (0/1).
  - Date-related features (`searchDate`, `flightDate`) were transformed into datetime format for **time-based calculations**.
  - ISO 8601 time format (`segmentsDepartureTimeRaw`, `segmentsArrivalTimeRaw`) was converted into **total minutes** for improved interpretability.
- **Segment-Based Features**:
  - Flight segment details (e.g., departure/arrival times, airline, and layovers) were initially stored as concatenated strings separated by `||`.
  - These were **expanded into separate columns** to allow for **granular-level feature extraction**.

---

## **Data Normalization and Encoding**
To ensure consistency and improve model performance:
- **Numerical Feature Scaling**:
  - Applied **Min-Max Scaling** to normalize continuous variables for model compatibility.
- **Categorical Encoding**:
  - **One-hot encoding** was applied to **nominal categorical features** (e.g., `destinationAirport`).
  - **Ordinal encoding** was used for features with an inherent ranking (e.g., `cabinClassesCombinations` to maintain seat class hierarchy).

---

## **Outlier Handling**
- Airfare pricing includes extreme values (e.g., last-minute bookings, premium tickets), requiring careful outlier handling.
- **Different Outlier Strategies for Different Models**:
  - For **Linear Regression** and **LSTM**, **Interquartile Range (IQR) capping** was applied at multiple thresholds.
  - Both **capped and uncapped datasets** were tested to evaluate the impact on preserving meaningful pricing variations.
  - **Random Forest**, being more robust to outliers, was tested on the original dataset without capping.

---

## **Feature Engineering and Enrichment**
To improve predictive accuracy, several new features were engineered:

| Feature Name | Type | Description |
|-------------|------|-------------|
| `daysToDeparture` | Numeric | Days between booking and flight date |
| `pricePerMile` | Numeric | Normalized total fare per travel distance |
| `isHoliday` | Categorical (0/1) | Whether the flight occurs on a major U.S. holiday |
| `totalLayoverTime` | Numeric | Sum of layover durations in minutes |
| `numberLegs` | Numeric | Number of flight segments (1 = direct, 2+ = layovers) |
| `fareLag_1` | Numeric | Fare from 1 day before booking |
| `fareLag_7` | Numeric | Fare from 7 days before booking |
| `avgFareLag` | Numeric | Average fare over past 7 days |

**Feature Enrichment**:
- **Holiday Indicators**: Flights were labeled if they occurred on or near major U.S. holidays (e.g., July 4th).
- **Layover Features**: Number of segments and total layover time were extracted to understand their impact on ticket prices.
- **Short-term Fare Trends**: Lag features were introduced to capture how fares change over time.

---

## **Bias & Limitations**
- **Bias**: Focused **only on LAX departures**, findings may not generalize to all U.S. airports.
- **Temporal Limitations**:  
  - Data only includes flights from **April – October 2022**.  
  - Our filtered data focuses on flights from **June - August 2022**.
  - Seasonal trends outside this range (e.g., winter fares) **are not represented**.

---

## **Usage License**
- **License**: [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Expected Update Frequency**: **Never (Static Dataset)**.

---

## **References**
- [Google’s Data Cards](https://modelcards.withgoogle.com/about)
- [Microsoft’s Datasheets for Datasets](https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RE4t8QB)
- [Dataset Source](https://www.kaggle.com/datasets/dilwong/flightprices)


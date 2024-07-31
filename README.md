# S&P 500 Stock Price Prediction

This project aims to predict the direction of the S&P 500 index by leveraging historical data and applying machine learning techniques. The primary objective is to determine whether the S&P 500 index will close higher or lower on the following day using a Random Forest Classifier. The project involves data acquisition, preprocessing, feature engineering, model training, evaluation, and backtesting.

## Project Overview

The project workflow consists of the following key steps:

1. **Data Acquisition**: Downloading historical S&P 500 data.
2. **Data Preprocessing**: Cleaning and preparing the data for analysis.
3. **Model Training and Evaluation**: Building and evaluating the predictive model.
4. **Backtesting**: Validating the model over historical periods.

## Prerequisites

To run this project, you need the following:

- Python 3
- Required Python packages:
  - `yfinance`: For downloading financial data.
  - `pandas`: For data manipulation and analysis.
  - `scikit-learn`: For machine learning algorithms and model evaluation.
  - `matplotlib`: For plotting and visualizations.

```

## Project Structure

- **`sp500_data.csv`**: A CSV file containing the historical S&P 500 data.
- **`stock_price_prediction.py`**: The main Python script that implements the data processing, model training, and prediction logic.
- **`README.md`**: This documentation file.

## Data Acquisition

The historical data for the S&P 500 index is obtained using the `yfinance` library. The data includes daily prices and volume, spanning the maximum available period. If the data file (`sp500_data.csv`) exists, it loads the data; otherwise, it downloads the data and saves it to a CSV file for future use.

## Data Preprocessing

1. **Data Loading**:
   - The script loads the S&P 500 data and sets the index to the date column.

2. **Data Cleaning**:
   - Unnecessary columns such as "Dividends" and "Stock Splits" are removed.
   - The "Tomorrow" column is created to represent the closing price of the next day.
   - The "Target" column is added, which indicates whether the next day's closing price is higher (1) or lower (0) than the current day's closing price.

3. **Data Filtering**:
   - The dataset is filtered to include data starting from January 1, 1990, to ensure a consistent time range for model training.


## Model Training and Evaluation

1. **Data Splitting**:
   - The data is split into training and testing sets. The last 100 days are used for testing, while the rest is used for training.

2. **Model Initialization**:
   - A Random Forest Classifier is initialized with parameters `n_estimators=200` and `min_samples_split=50`.

3. **Model Training**:
   - The model is trained on the training set using the defined predictors (features).

4. **Prediction and Evaluation**:
   - Predictions are made on the test set, and the precision score is calculated to evaluate the model's performance.
   - The results, including the actual and predicted values, are plotted for visual inspection.

## Backtesting

Backtesting is conducted to evaluate the model's performance over different historical periods. The process involves iterating over the dataset with a sliding window, training the model on past data, and predicting future data. The predictions and actual values are then analyzed to assess the model's consistency and accuracy.

## Results

- **Precision Score**: The precision score is calculated for the test set and backtest predictions to measure the accuracy of the model's positive predictions.
- **Prediction Distribution**: The distribution of the predicted and actual target values is analyzed.
- **Visualizations**: Plots of predictions vs actual targets and feature trends are provided.

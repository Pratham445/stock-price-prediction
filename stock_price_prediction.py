import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load S&P 500 data
if os.path.exists("sp500_data.csv"):
    sp500_data = pd.read_csv("sp500_data.csv", index_col=0)
else:
    sp500_data = yf.Ticker("^GSPC").history(period="max")
    sp500_data.to_csv("sp500_data.csv")

sp500_data.index = pd.to_datetime(sp500_data.index)

# Basic plot of closing prices
sp500_data.plot.line(y="Close", use_index=True)

# Remove unnecessary columns
sp500_data.drop(columns=["Dividends", "Stock Splits"], inplace=True)

# Add 'Tomorrow' and 'Target' columns
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)
sp500_data = sp500_data.loc["1990-01-01":].copy()

# Define predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Split data into training and testing sets
train_data = sp500_data.iloc[:-100]
test_data = sp500_data.iloc[-100:]

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
rf_model.fit(train_data[predictors], train_data["Target"])

# Make predictions and evaluate precision score
test_predictions = rf_model.predict(test_data[predictors])
test_predictions = pd.Series(test_predictions, index=test_data.index)
print("Initial Precision Score:", precision_score(test_data["Target"], test_predictions))

# Plot predictions vs actual targets
combined_results = pd.concat([test_data["Target"], test_predictions], axis=1)
combined_results.plot()

# Define the predict function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    predictions = model.predict(test[predictors])
    predictions = pd.Series(predictions, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], predictions], axis=1)
    return combined

# Define the backtest function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train_subset = data.iloc[0:i].copy()
        test_subset = data.iloc[i:(i+step)].copy()
        predictions = predict(train_subset, test_subset, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Perform backtesting
predictions = backtest(sp500_data, rf_model, predictors)
print("Predictions Value Counts:", predictions["Predictions"].value_counts())
print("Backtest Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))
print("Target Value Counts:", predictions["Target"].value_counts() / predictions.shape[0])

# Drop rows with NaN values
sp500_data.dropna(subset=[col for col in sp500_data.columns if col != "Tomorrow"], inplace=True)

# Update model with new hyperparameters
rf_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Perform backtesting with original predictors
predictions = backtest(sp500_data, rf_model, predictors)
print("Predictions Value Counts with Updated Model:", predictions["Predictions"].value_counts())
print("Backtest Precision Score with Updated Model:", precision_score(predictions["Target"], predictions["Predictions"]))
print("Target Value Counts with Updated Model:", predictions["Target"].value_counts() / predictions.shape[0])

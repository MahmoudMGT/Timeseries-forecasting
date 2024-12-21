# Timeseries-forecasting
#📈 Stock Price Forecasting with LSTM 📈

# Bitcoin Price Prediction using LSTM

This project uses an LSTM (Long Short-Term Memory) recurrent neural network to predict the future price of Bitcoin. The project involves the following steps:

## 1. Data Acquisition and Preparation

* **Install Libraries:** Installs necessary libraries like `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow` and `scikit-learn`.
* **Fetch Data:** Downloads historical Bitcoin price data from Yahoo Finance using the `yfinance` library.  The code specifies 'BTC-USD' as the ticker symbol. Adjust if necessary.  It downloads data for the past two years.
* **Data Exploration:** Prints the first few rows and info of the data for initial inspection.
* **Moving Averages:** Calculates 9, 21, and 80 day moving averages of the adjusted close price and plots them alongside the adjusted closing price to visualize price trends.
* **Fibonacci Retracement Levels:** Identifies the highest and lowest prices of the current year and calculates Fibonacci retracement levels. These levels are then plotted on the price chart.
* **Monthly Returns:** Calculates and displays the average monthly return on Bitcoin based on adjusted close prices.

## 2. LSTM Model Building and Training

* **Data Scaling:** Uses `MinMaxScaler` to normalize Bitcoin adjusted close price data to a range between 0 and 1.
* **Data Splitting:** Splits the data into training and testing sets (70% training, 30% testing).
* **Dataset Creation:** Creates sequences of past price data (`look_back` period) as input features (X) and the subsequent price as the target (Y) for the LSTM model.
* **Model Architecture:** Defines a sequential LSTM model using TensorFlow/Keras. The model has two LSTM layers with dropout for regularization, followed by a dense output layer.
* **Model Compilation:** Compiles the model using the Adam optimizer and mean squared error loss function.
* **Model Training:** Trains the model on the prepared training data for 80 epochs.  Includes validation data to monitor performance on the test set.
* **Model Summary:** Prints a summary of the model's architecture.
* **Performance Evaluation**:
    - Plots the training and validation loss over epochs to monitor training progress and potential overfitting.
    - Calculates and prints the Root Mean Squared Error (RMSE) on both training and testing data, quantifying the model’s prediction accuracy.
* **Prediction Visualization**: Generates a plot comparing the actual and predicted prices for the test set to visualize model performance.

## 3. Prediction

* **Predicts price:** Makes a prediction for the next day's price using the last `look_back` days of data. Prints the predicted price.


## Usage

1. **Setup:** Make sure you have Python and the necessary libraries installed. You can install them using `pip install -r requirements.txt` (create a `requirements.txt` file listing the required packages).
2. **Run the code:** Execute the Python script.  
3. **Interpret Results:** Examine the plots, RMSE values, and the predicted price for insights into Bitcoin price trends.

## Potential Improvements

* **Hyperparameter Tuning:** Experiment with different LSTM layer sizes, dropout rates, epochs, batch size, and `look_back` parameters to optimize the model's performance.
* **Feature Engineering:** Consider incorporating additional features, like trading volume, market sentiment, or other relevant indicators, to improve prediction accuracy.
* **More Robust Validation:** Implement more thorough validation techniques, like cross-validation, to ensure the model generalizes well to unseen data.
* **Alternative Models:** Explore other time series models, like ARIMA or Prophet, for comparison.


![fibo gold](https://github.com/user-attachments/assets/c1c85a06-1346-456f-9f92-64c26cecd710)

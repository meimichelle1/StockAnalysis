# Stock Price Predictor
This project aims to predict the stock price movement (increase or decrease) based on top news headlines and historical stock prices. The project involves data preprocessing, sentiment analysis, and machine learning models to make the predictions.
## Installation 
- Clone the repository or download the StockPricePredictor.ipynb file.
- Install the required libraries.
## Data
- Stock headlines data from Stock Headlines.csv.
- Historical stock prices data from p.csv.
## Sentiment Analysis
- Use TextBlob to compute subjectivity and polarity scores for combined headlines.
- Use VADER SentimentIntensityAnalyzer to get sentiment scores (compound, negative, neutral, positive).
## Model Training and Evaluation 
1. Support Vector Machines (SVM)
  - Train an SVM model using the features and target variable.
  - Evaluate the model using classification metrics.
2. Linear Discriminant Analysis (LDA)
- Train an LDA model using the features and target variable.
- Evaluate the model using classification metrics.
## Time Series Prediction with RNN 
1. Data Preparation:
   - Calculate stock returns.
   - Normalize the data.
   - Create datasets for time series prediction
2. LSTM Model:
- Build and train an LSTM model for autoregressive prediction.
- Visualize training loss and validation loss.
- Perform one-step and multi-step forecasts.
## Stock Price Classification 
1. Data Preparation
- Use Open, High, Low, Close prices as input features.
- Normalize the data and create training and testing datasets.
2. LSTM Model for Classification
- Build and train an LSTM model to predict whether the stock price will go up or down.
- Visualize training and validation loss and accuracy.

## Results
- The project demonstrates different approaches to predict stock price movements using sentiment analysis and machine learning models.
- It highlights the effectiveness of using sentiment scores combined with historical prices for making predictions.
- The best results are obtained using all columns for binary classification of stock price movement.

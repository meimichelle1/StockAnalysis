# Stock Prediction Project Overview
The stock prediction project can be split into two parts: Stock Price Analysis with Big Data and Stock Prediction with Sentiment Analysis. The first part uses only stock prices as input, while the second part uses both stock prices and related stock news for analysis.

## Stock Price Analysis with Big Data
This part focuses on performing basic analysis on the stock price data using PySpark for distributed processing.

### Apache Spark (PySpark)
PySpark efficiently handles large-scale data by distributing computations across clusters. As stock data grows over time, Spark allows for scalable and fast data processing. It loads and cleans the stock data by removing unnecessary symbols, like commas, and ensuring the data is in the correct format.

### Spark SQL
Spark SQL provides the ability to manipulate and query structured data using SQL-like operations. It simplifies tasks such as data cleaning, type casting, and performing aggregations, like computing average, minimum, and maximum stock prices across different dates. This provides useful insights into stock trends.

### Big Data Aggregation
Spark can group data by date and perform operations, such as calculating the average, minimum, and maximum stock prices across the entire dataset, making it easier to analyze trends.

### CSV Output
The processed DataFrames are saved to CSV files, allowing the results of the analysis to be easily shared and reused.

## Stock Prediction with Sentiment Analysis
This part involves using both stock price data and news sentiment data to predict future stock prices. The goal is to apply sentiment analysis to news headlines, integrate the resulting data with stock prices, and use machine learning (specifically linear regression) to predict future prices.

### Apache Spark (PySpark)
PySpark allows for efficient merging and processing of two datasets—stock prices and news. Spark joins the stock price data with the news data on the Date column, ensuring that both price and sentiment data are aligned for the same day.

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
VADER performs sentiment analysis by assigning a sentiment score (positive, neutral, or negative) to blocks of text, such as news headlines. News sentiment can have a significant impact on stock prices. VADER provides an easy way to numerically quantify the sentiment of news headlines, which can then be fed into the machine learning model.

### Spark SQL
Functions like concat_ws are used to combine multiple text columns (news headlines) into a single column for sentiment analysis. This generates a single column that contains all news headlines for a given day, which can then be passed to VADER for sentiment scoring.

### VectorAssembler
This PySpark feature is used to combine multiple input columns (e.g., Open, High, Low, and Sentiment) into a single vector, which is then used as input for machine learning models. In machine learning, all input features need to be combined into a feature vector before being fed into the model.

### Linear Regression
Linear regression is a machine learning algorithm that models the relationship between a dependent variable (stock price) and one or more independent variables (such as stock prices and sentiment). Linear regression is a common and effective technique for predicting numerical values like stock prices based on historical data.

### RegressionEvaluator
This utility evaluates the performance of regression models using metrics like Root Mean Squared Error (RMSE). It helps assess the accuracy of the stock price prediction model by measuring how close the predicted values are to the actual values.

### CSV Output
Once predictions are made, the DataFrames are saved to CSV files, allowing for further analysis or sharing of results.

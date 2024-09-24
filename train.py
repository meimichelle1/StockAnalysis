from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max, regexp_replace, to_date, concat_ws
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Spark session
spark = SparkSession.builder.appName("StockPredictionWithSentiment").getOrCreate()

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# UDF for sentiment analysis
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return float(scores['compound'])

sentiment_udf = udf(analyze_sentiment, FloatType())

def load_and_preprocess_stock_data(file_path):
    # Load stock data
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Clean and cast stock prices, also converting Date to correct format
    df = df.withColumn("Open", regexp_replace("Open", ",", "").cast(FloatType())) \
           .withColumn("High", regexp_replace("High", ",", "").cast(FloatType())) \
           .withColumn("Low", regexp_replace("Low", ",", "").cast(FloatType())) \
           .withColumn("Close", regexp_replace("Close", ",", "").cast(FloatType())) \
           .withColumn("Date", to_date(col("Date"), "MM/dd/yyyy"))  # Adjust format as per your dataset

    # Filter nulls
    df = df.filter(col("Close").isNotNull())
    return df

def load_and_preprocess_news_data(file_path):
    # Load news data
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Combine all news columns into a single 'Combined_News' column
    news_cols = [col(f"Top{i}") for i in range(1, 26)]  # Assuming news columns are named Top1, Top2, ..., Top25
    df = df.withColumn("Combined_News", concat_ws(" ", *news_cols)) \
           .withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))

    return df

def merge_stock_and_news(stock_df, news_df):
    # Merge on 'Date' column
    merged_df = stock_df.join(news_df, on="Date", how="inner")
    return merged_df

def perform_sentiment_analysis(df):
    # Apply sentiment analysis on 'Combined_News' column
    df = df.withColumn("Sentiment", sentiment_udf(col("Combined_News")))
    return df

def create_features(df):
    # Create feature vectors using 'Open', 'High', 'Low', and 'Sentiment'
    assembler = VectorAssembler(inputCols=["Open", "High", "Low", "Sentiment"], outputCol="features")
    return assembler.transform(df)

def build_and_train_model(df):
    # Use 'Close' as the label (target value)
    df = df.withColumn("label", col("Close"))

    # Split the data into training and testing sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Ensure train and test data are not empty
    print(f"Training dataset count: {train_data.count()}")
    print(f"Test dataset count: {test_data.count()}")

    if train_data.count() == 0 or test_data.count() == 0:
        print("Training dataset is empty! Exiting...")
        return None

    # Build and train the Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_data)

    # Make predictions on the test data
    predictions = lr_model.transform(test_data)

    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

    return predictions

def save_result_to_file(df, output_path):
    # Drop the 'features' column before saving to CSV
    df = df.drop("features")  # Drop the complex 'features' column
    df.write.mode("overwrite").csv(output_path, header=True)

if __name__ == "__main__":
    # File paths to stock and news data
    stock_file_path = "stock_price.csv"
    news_file_path = "stock_news.csv"

    print("Loading and preprocessing stock data...")
    stock_df = load_and_preprocess_stock_data(stock_file_path)
    stock_df.show(5)
    print(f"Total records in stock data: {stock_df.count()}")

    print("Loading and preprocessing news data...")
    news_df = load_and_preprocess_news_data(news_file_path)
    news_df.show(5)
    print(f"Total records in news data: {news_df.count()}")

    print("Merging stock and news data...")
    merged_df = merge_stock_and_news(stock_df, news_df)
    print(f"Total records after merging: {merged_df.count()}")
    merged_df.show(5)

    print("Performing sentiment analysis...")
    merged_with_sentiment_df = perform_sentiment_analysis(merged_df)
    merged_with_sentiment_df.show(5)
    print(f"Total records after sentiment analysis: {merged_with_sentiment_df.count()}")

    print("Creating features for the model...")
    feature_df = create_features(merged_with_sentiment_df)
    feature_df.show(5)
    print(f"Total records in feature data: {feature_df.count()}")

    print("Building and training the model...")
    predictions_df = build_and_train_model(feature_df)

    if predictions_df is not None:
        predictions_df.show(5)
        save_result_to_file(predictions_df, "stock_predictions_with_news_output.csv")
        print("Predictions saved successfully!")
    else:
        print("No predictions were made.")

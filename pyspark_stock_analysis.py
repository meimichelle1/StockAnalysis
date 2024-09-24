from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max
from pyspark.sql.types import FloatType
from pyspark.sql.functions import regexp_replace

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StockSentimentAnalysis") \
    .getOrCreate()

def load_and_preprocess(file_path):
    # Load data into a Spark DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Remove commas from the Open, High, Low, and Close columns and cast them to float
    df = df.withColumn("Open", regexp_replace("Open", ",", "").cast(FloatType())) \
           .withColumn("High", regexp_replace("High", ",", "").cast(FloatType())) \
           .withColumn("Low", regexp_replace("Low", ",", "").cast(FloatType())) \
           .withColumn("Close", regexp_replace("Close", ",", "").cast(FloatType()))

    # Filter out rows with null values in 'Close'
    df = df.filter(col("Close").isNotNull())

    return df

def analyze_stock_prices(df, aggregation="avg", field="Close"):
    # Perform aggregation (average, min, or max)
    if aggregation == "avg":
        result = df.groupBy("Date").agg(avg(col(field)).alias("Avg_" + field))
    elif aggregation == "min":
        result = df.groupBy("Date").agg(min(col(field)).alias("Min_" + field))
    elif aggregation == "max":
        result = df.groupBy("Date").agg(max(col(field)).alias("Max_" + field))
    else:
        raise ValueError("Invalid aggregation type. Choose from ['avg', 'min', 'max']")

    return result

def save_result_to_file(result, output_path):
    """
    Save the result of the stock analysis to a CSV file.
    Set the mode to overwrite if the file already exists.
    """
    result.write.mode("overwrite").csv(output_path, header=True)

if __name__ == "__main__":
    # Specify the path to your stock data CSV file
    file_path = "stock_price.csv"  # Replace with your actual file path
    
    # Load and preprocess stock data
    df = load_and_preprocess(file_path)

    # Perform big data analysis (e.g., average of the 'Close' field)
    result = analyze_stock_prices(df, aggregation="avg", field="Close")

    # Show the result in the console
    result.show()

    # Save the result to a CSV file
    output_path = "output_stock_analysis.csv"
    save_result_to_file(result, output_path)

    print(f"Stock analysis results have been saved to {output_path}")

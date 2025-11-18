# --------------------------------------------------------------------------
# Importing Required Libraries
# --------------------------------------------------------------------------

import boto3  # AWS SDK for Python to interact with AWS services like S3
import pandas as pd  # Pandas for handling and analyzing structured data
import io  # Provides the in-memory stream to read/write data (used for S3 object body)


# --------------------------------------------------------------------------
# Global Configuration Class
# --------------------------------------------------------------------------
class GlobalConfig:
    """
    GlobalConfig sets up AWS S3 configuration and client connection.

    Attributes:
        S3_BUCKET (str): The name of the S3 bucket.
        S3_KEY (str): The file key (path/filename) within the S3 bucket.
        s3 (boto3.client): A low-level S3 client used for performing operations like get_object().
    """

    def __init__(self, bucket_name: str, file_name: str):
        """
        Constructor initializes the S3 client and stores bucket details.

        Args:
            bucket_name (str): The name of the S3 bucket to connect to.
            file_name (str): The name or key of the CSV file inside the bucket.
        """
        self.S3_BUCKET = bucket_name  # Store S3 bucket name
        self.S3_KEY = file_name  # Store S3 file key (CSV file path)
        self.s3 = boto3.client("s3")  # Create an S3 client using default AWS credentials
        

# --------------------------------------------------------------------------
# ReadDataFrame Class for Fetching and Loading CSV Data from S3
# --------------------------------------------------------------------------
class ReadDataFrame(GlobalConfig):
    """
    ReadDataFrame extends GlobalConfig to provide functionality
    for reading a CSV file stored in an AWS S3 bucket into a Pandas DataFrame.
    """

    def __init__(self, bucket_name: str, file_name: str):
        """
        Initialize the ReadDataFrame class and inherit S3 connection setup.

        Args:
            bucket_name (str): Name of the target S3 bucket.
            file_name (str): Key or name of the CSV file to be fetched.
        """
        super().__init__(bucket_name, file_name)  # Call parent class constructor

    def read_dataframe(self):
        """
        Reads a CSV file from the specified S3 bucket and loads it into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing CSV data if successful.
            str: Error message if the file cannot be found or loaded.
        """
        try:
            # Print the S3 location being accessed
            print(f"Fetching file from s3://{self.S3_BUCKET}/{self.S3_KEY}")

            # Retrieve the object (file) from S3
            obj = self.s3.get_object(Bucket=self.S3_BUCKET, Key=self.S3_KEY)

            # Read CSV data directly from the S3 object body
            # io.BytesIO() converts binary data into an in-memory file-like object
            df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["device_time"])

            # Print confirmation message with number of records loaded
            print(f"Loaded {len(df)} records from s3://{self.S3_BUCKET}/{self.S3_KEY}")

            # Return the DataFrame for further processing
            return df

        except self.s3.exceptions.NoSuchKey:
            # Handle case where the specified file key does not exist in the bucket
            return f"File not found: {self.S3_KEY}"

        except Exception as e:
            # Handle any unexpected exceptions (network, permissions, parsing errors, etc.)
            return f"Error loading CSV from S3: {e}"


# --------------------------------------------------------------------------
# Main Program Execution (Only Runs When Script is Executed Directly)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This block runs only when the script is executed directly.
    It will:
      1. Create an instance of ReadDataFrame with bucket and file details.
      2. Load the CSV from S3 into a DataFrame.
      3. Print the shape (rows, columns) of the loaded DataFrame.
    """

    # Create instance of the ReadDataFrame class with bucket and file details
    r = ReadDataFrame(bucket_name='estac-data', file_name='estac_data.csv')

    # Read the CSV file into a DataFrame
    data = r.read_dataframe()

    # Print the shape (number of rows, number of columns) of the DataFrame
    print(data.shape)

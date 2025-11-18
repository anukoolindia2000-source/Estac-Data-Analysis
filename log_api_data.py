import requests
import pandas as pd
import os
from datetime import datetime, timedelta


class FetchDate:
    """
    Stores the start and end date for API fetch.

    Methods
    -------
    set_date_range(days: int = 0):
        Sets START_DATE and END_DATE dynamically based on current date.
    """
    START_DATE = None
    END_DATE = None

    @classmethod
    def set_date_range(cls, days: int = 0):
        """
        Set START_DATE and END_DATE dynamically.

        Parameters
        ----------
        days : int
            Number of days back for START_DATE (default 0 = today)
        """
        cls.END_DATE = datetime.now()
        cls.START_DATE = cls.END_DATE - timedelta(days=days)
        print(f"Fetch Date set START: {cls.START_DATE.date()}, END: {cls.END_DATE.date()}")


class GlobalConfig:
    """
    Stores API configuration and dynamically builds the API URL with dates.

    Attributes
    ----------
    CSV_FILE : str
        CSV filename for storing data.
    API_KEY : str
        API key for authentication.
    API_URL : str
        Full API URL including start and end dates.
    """

    def __init__(self, file: str, key: str, url: str):
        self.CSV_FILE = file
        self.API_KEY = key
        # Format datetime objects as strings for URL
        self.API_URL = (
            f"{url}"
            f"start={FetchDate.START_DATE.strftime('%Y-%m-%d')}&"
            f"end={FetchDate.END_DATE.strftime('%Y-%m-%d')}"
        )

    def get_headers(self):
        """Return the headers required for API requests."""
        try:
            return {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
            }
        except Exception as e:
            return e


class APIClient:
    """Handles API communication."""
    def __init__(self, url: str, headers: dict):
        self.url = url
        self.headers = headers

    def fetch_data(self):
        """Fetch data from API endpoint and return as JSON."""
        try:
            print(f"Fetching data from API: {self.url}")
            response = requests.get(self.url, headers=self.headers, timeout=15)
            response.raise_for_status()
            print("Data successfully fetched from API.")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return None


class CSVManager:
    """Handles CSV file operations."""
    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    def load_existing_data(self):
        """Load existing CSV if available."""
        try:
            if os.path.exists(self.csv_file):
                print(f"Loading existing CSV: {self.csv_file}")
                return pd.read_csv(self.csv_file, parse_dates=["device_time"])
            print("No existing CSV found.")
            return None
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

    def append_new_data(self, df_new: pd.DataFrame):
        """Append new records to CSV."""
        try:
            df_new.to_csv(self.csv_file, mode="a", index=False, header=False)
            print(f"Appended {len(df_new)} new records to {self.csv_file}.")
        except Exception as e:
            print(f"Error appending data: {e}")

    def create_csv(self, df: pd.DataFrame):
        """Create CSV with initial data."""
        try:
            df.to_csv(self.csv_file, index=False)
            print(f"Created {self.csv_file} with {len(df)} records.")
        except Exception as e:
            print(f"Error creating CSV: {e}")


class DataProcessor:
    """Cleans and filters API data."""
    @staticmethod
    def clean_data(data):
        """Clean raw API JSON into a DataFrame."""
        try:
            df = pd.DataFrame(data)
            df["humidity"] = df["humidity"].astype(float)
            df["temperature"] = df["temperature"].astype(float)
            df["device_time"] = pd.to_datetime(df["device_time"])
            print(f"Data cleaned: {len(df)} records found.")
            return df
        except (KeyError, ValueError, TypeError) as e:
            print(f"Data cleaning error: {e}")
            return pd.DataFrame()

    @staticmethod
    def filter_new_records(df_api: pd.DataFrame, df_existing: pd.DataFrame):
        """Filter only new records not already in CSV."""
        try:
            last_time = df_existing["device_time"].max()
            print(f"Last record timestamp in CSV: {last_time}")
            return df_api[df_api["device_time"] > last_time]
        except KeyError as e:
            print(f"Filtering error: {e}")
            return pd.DataFrame()


class DataPipeline:
    """Orchestrates API fetch, cleaning, and CSV storage."""
    def __init__(self, config: GlobalConfig):
        self.api_client = APIClient(config.API_URL, config.get_headers())
        self.csv_manager = CSVManager(config.CSV_FILE)
        self.processor = DataProcessor()

    def run(self):
        """Run the full pipeline."""
        try:
            api_data = self.api_client.fetch_data()
            if not api_data:
                print("No data returned from API. Exiting.")
                return

            df_api = self.processor.clean_data(api_data)
            if df_api.empty:
                print("No valid data after cleaning. Exiting.")
                return

            df_existing = self.csv_manager.load_existing_data()
            if df_existing is not None:
                df_new = self.processor.filter_new_records(df_api, df_existing)
                if df_new.empty:
                    print("No new data to append. CSV is up to date.")
                else:
                    self.csv_manager.append_new_data(df_new)
            else:
                self.csv_manager.create_csv(df_api)
        except Exception as e:
            print(f"Unexpected error in pipeline: {e}")


# ======================
# Example Usage
# ======================
if __name__ == "__main__":
    # Set dynamic fetch dates (e.g., last 1 day)
    FetchDate.set_date_range(days=1)

    # Initialize configuration with dynamic values
    api_url = "https://api.nbsense.in/th_ms/line?meter_id=1602&sensor_name=161&"  # replace with actual API
    api_key = "e6197229-5e38-4ce9-ab8a-f0192adc7782"
    csv_file = "estac_data_.csv"

    config = GlobalConfig(file=csv_file, key=api_key, url=api_url)

    # Run the pipeline
    pipeline = DataPipeline(config)
    pipeline.run()

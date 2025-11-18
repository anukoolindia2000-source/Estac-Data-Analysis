"""
Data Pipeline: API → SQL Storage
--------------------------------

This script fetches data from an API, cleans it, and inserts it into a SQL database table.
It ensures that only new records (based on `device_time`) are inserted.

Components:
- FetchDate: Manages start and end dates for API fetch.
- GlobalConfig: Stores API/DB configuration and builds URLs.
- APIClient: Handles API calls.
- SQLManager: Creates table, reads last timestamp, and inserts data.
- DataProcessor: Cleans API data and filters new records.
- DataPipeline: Orchestrates everything together.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy import text, types as satypes


class FetchDate:
    """
    Manages the dynamic date range for fetching API data.
    """
    START_DATE = None
    END_DATE = None

    @classmethod
    def set_date_range(cls, days: int = 0):
        """
        Set the date range dynamically based on the current date.

        Parameters
        ----------
        days : int
            Number of days back for START_DATE (default 0 = today).
        """
        cls.END_DATE = datetime.now()
        cls.START_DATE = cls.END_DATE - timedelta(days=days)
        print(f"Fetch Date set START: {cls.START_DATE.date()}, END: {cls.END_DATE.date()}")


class GlobalConfig:
    """
    Stores global configuration for API and database.

    Attributes
    ----------
    DB_URL : str
        SQLAlchemy database connection string.
    TABLE_NAME : str
        Target SQL table.
    ENGINE : sqlalchemy.Engine
        SQLAlchemy engine for DB communication.
    API_KEY : str
        Authentication token.
    API_URL : str
        API URL with dynamic start/end dates.
    """
    def __init__(self, db_url: str, table: str, key: str, url: str):
        self.DB_URL = db_url
        self.TABLE_NAME = table
        self.ENGINE = sqlalchemy.create_engine(self.DB_URL, pool_pre_ping=True)

        self.API_KEY = key
        self.API_URL = (
            f"{url}"
            f"start={FetchDate.START_DATE.strftime('%Y-%m-%d')}&"
            f"end={FetchDate.END_DATE.strftime('%Y-%m-%d')}"
        )

    def get_headers(self):
        """Return request headers for API calls."""
        return {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }


class APIClient:
    """
    Handles HTTP requests to the API.
    """
    def __init__(self, url: str, headers: dict):
        self.url = url
        self.headers = headers

    def fetch_data(self):
        """
        Fetch data from API endpoint.

        Returns
        -------
        dict | list | None
            JSON data if successful, None if request fails.
        """
        try:
            print(f"Fetching data from API: {self.url}")
            r = requests.get(self.url, headers=self.headers, timeout=15)
            r.raise_for_status()
            print("Data successfully fetched from API.")
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return None


class SQLManager:
    """
    Manages SQL database operations.
    """
    def __init__(self, engine: sqlalchemy.Engine, table_name: str):
        self.engine = engine
        self.table_name = table_name

    def create_table_if_needed(self):
        """
        Create the table if it doesn’t already exist.
        Ensures schema for device_time, humidity, temperature.
        """
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            device_time DATETIME NOT NULL,
            humidity DECIMAL(10,3),
            temperature DECIMAL(10,3),
            UNIQUE KEY uk_{self.table_name}_device_time (device_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
        print(f"Ensured table exists: {self.table_name}")

    def get_last_device_time(self):
        """
        Fetch the most recent device_time from SQL.

        Returns
        -------
        datetime | None
            Last timestamp in SQL or None if table is empty.
        """
        q = text(f"SELECT MAX(device_time) AS last_time FROM {self.table_name}")
        with self.engine.connect() as conn:
            row = conn.execute(q).mappings().first()
            return row["last_time"] if row and row["last_time"] is not None else None

    def insert_dataframe(self, df: pd.DataFrame):
        """
        Insert cleaned dataframe into SQL table.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing new rows.
        """
        dtype = {
            "device_time": satypes.DateTime(),
            "humidity": satypes.Numeric(10, 3),
            "temperature": satypes.Numeric(10, 3),
        }
        df.to_sql(
            name=self.table_name,
            con=self.engine,
            index=False,
            if_exists="append",
            dtype=dtype
        )
        print(f"Inserted {len(df)} rows into {self.table_name}.")


class DataProcessor:
    """
    Cleans and filters API data.
    """
    @staticmethod
    def clean_data(data):
        """
        Convert raw JSON into a DataFrame with correct types.

        Parameters
        ----------
        data : dict | list
            API JSON response.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with datetime and numeric types.
        """
        try:
            df = pd.DataFrame(data)
            df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
            df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
            df["device_time"] = pd.to_datetime(df["device_time"], errors="coerce")
            df = df.dropna(subset=["device_time"])  # drop invalid rows
            print(f"Data cleaned: {len(df)} records found.")
            return df
        except (KeyError, ValueError, TypeError) as e:
            print(f"Data cleaning error: {e}")
            return pd.DataFrame()

    @staticmethod
    def filter_new_records(df_api: pd.DataFrame, last_time):
        """
        Filter only new records that are not already in SQL.

        Parameters
        ----------
        df_api : pd.DataFrame
            Fresh API data.
        last_time : datetime
            Last timestamp present in SQL.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only new records.
        """
        if last_time is None:
            print("No existing data in SQL. All API rows are new.")
            return df_api.sort_values("device_time")

        print(f"Last record timestamp in SQL: {last_time}")
        return df_api[df_api["device_time"] > pd.to_datetime(last_time)].sort_values("device_time")


class DataPipeline:
    """
    Orchestrates the data pipeline: API → Processing → SQL.
    """
    def __init__(self, config: GlobalConfig):
        self.api_client = APIClient(config.API_URL, config.get_headers())
        self.sql_manager = SQLManager(config.ENGINE, config.TABLE_NAME)
        self.processor = DataProcessor()

    def run(self):
        """
        Run the full pipeline:
        1. Ensure SQL table exists.
        2. Fetch API data.
        3. Clean and parse into DataFrame.
        4. Filter new rows.
        5. Insert into SQL.
        """
        try:
            self.sql_manager.create_table_if_needed()

            api_data = self.api_client.fetch_data()
            if not api_data:
                print("No data returned from API. Exiting.")
                return

            df_api = self.processor.clean_data(api_data)
            if df_api.empty:
                print("No valid data after cleaning. Exiting.")
                return

            last_time = self.sql_manager.get_last_device_time()
            df_new = self.processor.filter_new_records(df_api, last_time)

            if df_new.empty:
                print("No new data to insert. SQL is up to date.")
                return

            self.sql_manager.insert_dataframe(df_new)

        except Exception as e:
            print(f"Unexpected error in pipeline: {e}")


# ======================
# Example Usage
# ======================
if __name__ == "__main__":
    # 1) Set fetch window (last 1 day)
    FetchDate.set_date_range(days=1)

    # 2) DB Config (adjust credentials & DB name)
    DB_URL = "mysql+pymysql://root:anukool@localhost:3306/godrej_estac"
    TABLE = "estac_data"

    # 3) API Config
    api_url = "https://api.nbsense.in/th_ms/line?meter_id=1602&sensor_name=161&"
    api_key = "e6197229-5e38-4ce9-ab8a-f0192adc7782"

    config = GlobalConfig(db_url=DB_URL, table=TABLE, key=api_key, url=api_url)

    # 4) Run pipeline
    pipeline = DataPipeline(config)
    pipeline.run()

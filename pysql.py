import os
import pymysql
import sqlalchemy
import pandas as pd

# Class for connecting to a MySQL database
class Database_connect:
    
    # Initialize the class with password and database name
    def __init__(self, password, database):
        self.password = password
        self.database_name = database

    # Method to try establishing a database connection
    def try_connection(self):
        try:
            # Create SQLAlchemy engine and establish a connection
            self.engine = sqlalchemy.create_engine(
                f'mysql+pymysql://root:{self.password}@localhost:3306/{self.database_name}')
            self.conn = pymysql.connect(
                host='localhost',
                user='root',
                password=self.password,
                db=self.database_name,
            )

            # Return cursor if connection is successful
            return self.conn.cursor()
        except:
            return None

    # Method to select all records from a table
    def select_tables(self, table_name):
        try:
            # Attempt to establish a connection
            self.cur = self.try_connection()
            # Execute query to select all records from the specified table
            self.cur.execute(f"select * from {table_name}")
            # Fetch all records
            output = self.cur.fetchall()
            # Close cursor
            self.cur.close()
            return output  # Return the selected records
        except:
            return -1  # Return -1 if an error occurs

    # Method to create a table from a CSV file
    def create_table(self, file, table_name):
        if os.path.isfile(file):  # Check if the file exists
            self.df = pd.read_csv(file)  # Read CSV file into a DataFrame
            try:
                # Attempt to establish a connection
                self.cur = self.try_connection()
                # Execute query to create the table
                self.cur.execute(f"create table {table_name}(name varchar(30))")
                # Iterate over columns in DataFrame
                for cols in self.df.columns:
                    if self.df[cols].dtypes != 'object':
                        # If column type is not object, add as int
                        self.cur.execute(f"alter table {table_name} add({cols} int)")
                    else:
                        # If column type is object, add as varchar
                        self.cur.execute(f"alter table {table_name} add({cols} varchar(100))")
                # Drop 'name' column from the table
                self.cur.execute(f"alter table {table_name} drop name")

                # Write DataFrame to SQL table
                self.df.to_sql(
                    name=table_name,
                    con=self.engine,
                    index=False,
                    if_exists='append'
                )
            except:
                return -1  # Return -1 if an error occurs
        else:
            return -1  # Return -1 if the file doesn't exist

if __name__ == "__main__":
   

   db = Database_connect(password='anukool',database='godrej_estac')
   db.create_table(file='D:/Estac Data Analysis/estac_records.csv',table_name='estac_data')
import mysql.connector
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()


def parse_fits_data(fits_data: bytes, dtype: str, height: int, width: int):
    """Parse FITS data from bytes to a numpy array."""
    return np.frombuffer(fits_data, dtype=dtype).reshape((height, width))


def connect_to_database():
    # Connect to server
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 3306),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def test_connection():
    try:
        conn = connect_to_database()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    else:
        cur = conn.cursor()
        cur.execute("SELECT CURDATE()")
        # Fetch one result
        row = cur.fetchone()
        print("Current date is: {0}".format(row[0]))
        # Close cursor
        cur.close()
        # Close connection
        conn.close()


def load_fits_records():
    """Load FITS file metadata from the database."""
    conn = connect_to_database()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM fits_files")
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return [ record |
       {"array": parse_fits_data(
            record["fits_data"],
            dtype=record["dtype"],
            height=record["height"],
            width=record["width"],
        )}
        for record in results
    ]


def insert_fits_data(data: dict):
    """Insert FITS file and metadata into the database."""
    conn = connect_to_database()
    cursor = conn.cursor()

    placeholders = ", ".join(["%s"] * len(data))
    columns = ", ".join(data.keys())
    sql = f"INSERT INTO fits_files ({columns}) VALUES ({placeholders})"

    print(f"Executing SQL: {sql}")

    cursor.execute(sql, list(data.values()))
    conn.commit()

    cursor.close()
    conn.close()



```python
import os
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Database configuration
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    try:
        yield conn
    finally:
        conn.close()

class EngagementMetrics:
    def __init__(self):
        self.create_table()

    def create_table(self):
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS engagement_metrics (
                        id SERIAL PRIMARY KEY,
                        user_id INT NOT NULL,
                        follows INT DEFAULT 0,
                        comments INT DEFAULT 0,
                        likes INT DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    def record_engagement(self, user_id, follows=0, comments=0, likes=0):
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Invalid user_id. Must be a positive integer.")
        if not all(isinstance(x, int) and x >= 0 for x in [follows, comments, likes]):
            raise ValueError("Engagement metrics must be non-negative integers.")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO engagement_metrics (user_id, follows, comments, likes)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, follows, comments, likes))
                conn.commit()

    def get_engagement_data(self, user_id):
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Invalid user_id. Must be a positive integer.")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT follows, comments, likes, created_at
                    FROM engagement_metrics
                    WHERE user_id = %s
                """, (user_id,))
                return cur.fetchall()

    def get_analytics(self):
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, SUM(follows) as total_follows, SUM(comments) as total_comments, SUM(likes) as total_likes
                    FROM engagement_metrics
                    GROUP BY user_id
                """)
                return cur.fetchall()

# Testing
import unittest
from unittest.mock import patch

class TestEngagementMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = EngagementMetrics()

    @patch('psycopg2.connect')
    def test_record_engagement(self, mock_connect):
        mock_connect.return_value.cursor.return_value.__enter__.return_value.execute.return_value = None
        self.metrics.record_engagement(1, follows=5, comments=3, likes=10)
        mock_connect.return_value.cursor.return_value.__enter__.return_value.execute.assert_called()

    @patch('psycopg2.connect')
    def test_get_engagement_data(self, mock_connect):
        mock_connect.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = [(5, 3, 10, '2023-10-01')]
        data = self.metrics.get_engagement_data(1)
        self.assertEqual(data, [(5, 3, 10, '2023-10-01')])

    @patch('psycopg2.connect')
    def test_get_analytics(self, mock_connect):
        mock_connect.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = [(1, 5, 3, 10)]
        analytics = self.metrics.get_analytics()
        self.assertEqual(analytics, [(1, 5, 3, 10)])

if __name__ == '__main__':
    unittest.main()
```

This code includes input validation, error handling, improved database schema, analytics methods, and comprehensive testing. The database configuration is moved to environment variables for security. The code is modular and testable, following best practices in software design.
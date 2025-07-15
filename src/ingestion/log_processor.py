import pandas as pd
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_logs_into_sessions(file_path: str, session_timeout_minutes: int = 30) -> List[List[Dict]]:
    """
    Processes web server logs into user sessions.

    A user session is defined as a sequence of log entries from the same IP address
    where the time between consecutive entries does not exceed the session timeout.

    Args:
        file_path (str): The path to the log file.
        session_timeout_minutes (int): The timeout in minutes to define a session break.

    Returns:
        List[List[Dict]]: A list of sessions, where each session is a list of log entry dictionaries.
    """
    try:
        logging.info(f"Processing log file: {file_path}")
        
        # Define column names as the file doesn't have a header
        # The last column is optional (e.g., 'add_to_cart'), so we'll handle it flexibly.
        col_names = ['product_name', 'category', 'timestamp', 'month', 'day_of_week', 
                     'department', 'ip_address', 'url', 'action']
        
        df = pd.read_csv(file_path, header=None, names=col_names, encoding='latin1', on_bad_lines='warn')

        # Data Cleaning and Preparation
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp', 'ip_address'], inplace=True)
        df.sort_values(by=['ip_address', 'timestamp'], inplace=True)
        
        # Calculate time difference between consecutive events for each user
        df['time_diff'] = df.groupby('ip_address')['timestamp'].diff().dt.total_seconds() / 60
        
        # Identify session starts
        # A new session starts if the time difference is greater than the timeout or if it's the first event for a user
        df['session_start'] = (df['time_diff'] > session_timeout_minutes) | (df['time_diff'].isnull())
        df['session_id'] = df['session_start'].cumsum()

        logging.info(f"Identified {df['session_id'].nunique()} sessions.")

        # Group by session_id to create the list of sessions
        sessions = df.groupby('session_id').apply(lambda x: x.to_dict('records')).tolist()
        
        logging.info("Successfully processed logs into sessions.")
        return sessions

    except FileNotFoundError:
        logging.error(f"Log file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during log processing: {e}")
        raise 
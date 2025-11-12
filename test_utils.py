import requests
import json
import time
import subprocess
import sys
import os

# --- Import db instance for direct access ---
from database import db

# --- Configuration ---
# Make the API URL configurable, with a sensible default for local testing
API_BASE_URL = os.environ.get("PLTV_API_BASE_URL", "http://127.0.0.1:5000")
RETRAIN_SECRET_KEY = os.environ.get("RETRAIN_SECRET_KEY", "YOUR_SECRET_KEY")

# --- Test Helper Functions ---

def clear_database():
    """Clears the database directly using the db instance."""
    print("\n--- Clearing Database Directly ---")
    try:
        # Ensure .env is loaded if script is run in a context where it hasn't been
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            
        db.clear_all_tables()
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Database clearing failed: {e}", file=sys.stderr)
        raise RuntimeError("Database clearing failed. Aborting tests.") from e

def send_event(customer_id, event_name, event_data={}):
    """
    Sends a single event to the /event endpoint, simulating the GA4 event structure.
    The API expects a payload that can be a single event or a list of events.
    To align with the sGTM GA4 tag, we send a structure containing an 'events' list.
    """
    # This is the actual event payload, mimicking a GA4 event
    event_payload = {
        "event_name": event_name,
        "client_id": customer_id,  # Use client_id to simulate the GA4 field
        "timestamp_micros": int(time.time() * 1_000_000),
        **event_data
    }
    
    # The API endpoint is designed to handle a batch of events in a list
    final_payload = {"events": [event_payload]}

    try:
        response = requests.post(f"{API_BASE_URL}/event", json=final_payload)
        response.raise_for_status()
        print(f"Sent event '{event_name}' for customer '{customer_id}'. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending event '{event_name}': {e}", file=sys.stderr)
        print("Please ensure the Flask API server is running.", file=sys.stderr)
        raise RuntimeError(f"Failed to send event '{event_name}'.") from e

def get_prediction(customer_id):
    """Gets a pLTV prediction for a customer."""
    print(f"\n--- Getting prediction for {customer_id} ---")
    payload = {"customer_id": customer_id}
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        if response.status_code == 404:
            print("Prediction: No data found for customer yet.")
            return None
        response.raise_for_status()
        return response.json() # Return the full JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error getting prediction: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to get prediction for customer '{customer_id}'.") from e

def trigger_retraining(wait_time=5):
    """Calls the /retrain endpoint and waits for it to likely complete."""
    print("\n--- Triggering Model Retraining ---")
    if RETRAIN_SECRET_KEY == "YOUR_SECRET_KEY":
        print("SKIPPING: RETRAIN_SECRET_KEY is not set. Cannot trigger retraining.", file=sys.stderr)
        return False
        
    url = f"{API_BASE_URL}/retrain?secret={RETRAIN_SECRET_KEY}"
    try:
        response = requests.post(url)
        response.raise_for_status()
        print("Retraining triggered successfully.")
        print(response.json().get("message"))
        if wait_time > 0:
            print(f"Waiting {wait_time} seconds for retraining to complete...")
            time.sleep(wait_time)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error triggering retraining: {e}", file=sys.stderr)
        if e.response:
            print(f"Response: {e.response.status_code} {e.response.text}", file=sys.stderr)
        return False

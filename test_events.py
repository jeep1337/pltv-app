
import requests
import json
import time
import subprocess
import sys
import os

API_BASE_URL = "http://127.0.0.1:5000"

def clear_database():
    """Executes the clear_db.py script to clear the database."""
    print("\n--- Clearing Database ---")
    try:
        # Construct path to clear_db.py relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        clear_db_path = os.path.join(script_dir, 'clear_db.py')

        result = subprocess.run([sys.executable, clear_db_path], capture_output=True, text=True, check=True)
        print("Database cleared successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to clear database.")
        print(e.stderr)
        sys.exit(1)

def send_event(customer_id, event_name, event_data={}):
    """Sends a single event to the /event endpoint."""
    payload = {
        "customer_id": customer_id,
        "event_data": {
            "events": [
                {
                    "event_name": event_name,
                    "timestamp_micros": int(time.time() * 1_000_000),
                    **event_data
                }
            ]
        }
    }
    try:
        response = requests.post(f"{API_BASE_URL}/event", json=payload)
        response.raise_for_status()
        print(f"Sent event '{event_name}' for customer '{customer_id}'. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending event '{event_name}': {e}")
        print("Please ensure the Flask API server is running.")
        sys.exit(1)


def validate_prediction(customer_id, expected_features):
    """Gets a pLTV prediction and validates the returned features."""
    print("\n--- Validating Prediction ---")
    payload = {"customer_id": customer_id}
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        response.raise_for_status()
        print(f"Prediction request for '{customer_id}' successful. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error getting prediction: {e}")
        sys.exit(1)

    data = response.json()
    features = data.get('features', {})
    pltv = data.get('pltv')

    print("\n--- Returned Data ---")
    print(f"pLTV: {pltv}")
    print("Features:", json.dumps(features, indent=2))
    
    print("\n--- Verification ---")
    validation_passed = True
    for key, expected_value in expected_features.items():
        actual_value = features.get(key)
        if actual_value != expected_value:
            validation_passed = False
            print(f"FAILED: Feature '{key}'. Expected: {expected_value}, Got: {actual_value}")
        else:
            print(f"PASSED: Feature '{key}'. Expected: {expected_value}, Got: {actual_value}")

    # Also validate the final pLTV value
    if pltv != expected_features['total_purchase_value']:
        validation_passed = False
        print(f"FAILED: pLTV value. Expected: {expected_features['total_purchase_value']}, Got: {pltv}")
    else:
        print(f"PASSED: pLTV value. Expected: {expected_features['total_purchase_value']}, Got: {pltv}")

    print("\n-------------------------")
    if validation_passed:
        print("✅ ✅ ✅ VALIDATION SUCCEEDED ✅ ✅ ✅")
    else:
        print("❌ ❌ ❌ VALIDATION FAILED ❌ ❌ ❌")
    print("-------------------------")


def main():
    """Main function to run the validation test."""
    # --- Test Configuration ---
    test_customer_id = "validation_customer_001"
    purchase_1_value = 100.0
    purchase_2_value = 50.0
    expected = {
        "number_of_purchases": 2,
        "total_purchase_value": purchase_1_value + purchase_2_value,
        "average_purchase_value": (purchase_1_value + purchase_2_value) / 2,
        "number_of_page_views": 3
    }

    # 1. Clear the database
    clear_database()

    # 2. Simulate the user journey
    print("\n--- Simulating User Journey ---")
    send_event(test_customer_id, "page_view")
    time.sleep(0.1)
    send_event(test_customer_id, "page_view")
    time.sleep(0.1)
    send_event(test_customer_id, "purchase", {"value": purchase_1_value})
    time.sleep(0.1) # Simulate time between events
    send_event(test_customer_id, "page_view")
    time.sleep(0.1)
    send_event(test_customer_id, "purchase", {"value": purchase_2_value})

    # 3. Get and validate the prediction
    validate_prediction(test_customer_id, expected)


if __name__ == "__main__":
    main()


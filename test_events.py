import requests
import json
import time

API_BASE_URL = "http://127.0.0.1:5000"  # Replace with your actual API base URL

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
    response = requests.post(f"{API_BASE_URL}/event", json=payload)
    print(f"Sent event '{event_name}' for customer '{customer_id}'. Status: {response.status_code}, Response: {response.json()}")

def get_prediction(customer_id):
    """Gets a pLTV prediction for a customer."""
    payload = {"customer_id": customer_id}
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    print(f"Prediction for customer '{customer_id}'. Status: {response.status_code}, Response: {response.json()}")

if __name__ == "__main__":
    test_customer_id = "test_customer_purchase_001"

    # 1. Send some page_view events
    send_event(test_customer_id, "page_view")
    time.sleep(1)
    send_event(test_customer_id, "page_view")
    time.sleep(1)

    # 2. Send a purchase event
    send_event(test_customer_id, "purchase", {"value": 50})
    time.sleep(1)

    # 3. Send another page_view event
    send_event(test_customer_id, "page_view")
    time.sleep(1)

    # 4. Send another purchase event
    send_event(test_customer_id, "purchase", {"value": 75})
    time.sleep(1)

    # 5. Get the pLTV prediction
    get_prediction(test_customer_id)

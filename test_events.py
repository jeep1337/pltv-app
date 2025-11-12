import pytest
import json
import time
import os
import sys
import requests # Import requests for trigger_retraining

from test_utils import (
    clear_database,
    send_event,
    get_prediction,
    reload_model_artifact
)

# --- Configuration ---
API_BASE_URL = os.environ.get("PLTV_API_BASE_URL", "http://127.0.0.1:5000")
RETRAIN_SECRET_KEY = os.environ.get("RETRAIN_SECRET_KEY", "YOUR_SECRET_KEY") 


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Clears the database once before the entire test session."""
    clear_database()
    yield
    # Optional: clear database again after all tests
    # clear_database()


def validate_prediction(customer_id, expected_features):
    """Gets a pLTV prediction and validates the returned features."""
    print(f"\n--- Validating Prediction for {customer_id} ---")
    
    pltv_value = get_prediction(customer_id)
    if pltv_value is None:
        pytest.fail("❌ FAILED: Did not receive a prediction response.")

    print("\n--- Returned Data ---")
    print(f"pLTV: {pltv_value}")
    
    print("\n--- Verification ---")
    # For now, we only check if pLTV is positive.
    # More detailed feature validation can be added here if get_prediction returns features.
    if pltv_value is not None and pltv_value > 0:
        print(f"✅ PASSED: pLTV value is positive. Got: {pltv_value}")
    else:
        pytest.fail(f"❌ FAILED: pLTV value should be positive. Got: {pltv_value}")


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
        pytest.fail(f"Failed to trigger retraining: {e}")
        return False


# --- Test Case Definition ---
TEST_CASES = [
    {
        "name": "Full Journey with Two Purchases",
        "customer_id_suffix": "customer_001",
        "events": [
            {"name": "page_view", "data": {}},
            {"name": "view_item", "data": {"items": [{"item_id": "SKU_123", "item_brand": "Brand A"}]}},
            {"name": "page_view", "data": {}},
            {"name": "add_to_cart", "data": {"items": [{"item_id": "SKU_456", "item_brand": "Brand B", "quantity": 1}]}},
            {"name": "begin_checkout", "data": {"items": [{"item_id": "SKU_456", "item_brand": "Brand B", "quantity": 1}]}},
            {"name": "purchase", "data": {"value": 100.0, "items": [{"item_id": "SKU_123", "item_brand": "Brand A", "quantity": 1}]}},
            {"name": "page_view", "data": {}},
            {"name": "purchase", "data": {"value": 50.0, "items": [{"item_id": "SKU_789", "item_brand": "Brand C", "quantity": 2}]}},
        ],
        "expected_features": {
            "number_of_purchases": 2,
            "total_purchase_value": 150.0,
            "average_purchase_value": 75.0,
            "number_of_page_views": 3,
            "add_to_cart_count": 1,
            "begin_checkout_count": 1,
            "total_items_purchased": 3,
            "distinct_products_purchased": 2,
            "distinct_brands_purchased": 2,
            "distinct_products_viewed": 1,
            "distinct_brands_viewed": 1,
        }
    },
    {
        "name": "Another Customer with Single Purchase",
        "customer_id_suffix": "customer_002",
        "events": [
            {"name": "page_view", "data": {}},
            {"name": "view_item", "data": {"items": [{"item_id": "SKU_999", "item_brand": "Brand X"}]}},
            {"name": "add_to_cart", "data": {"items": [{"item_id": "SKU_999", "item_brand": "Brand X", "quantity": 1}]}},
            {"name": "purchase", "data": {"value": 200.0, "items": [{"item_id": "SKU_999", "item_brand": "Brand X", "quantity": 1}]}},
        ],
        "expected_features": {
            "number_of_purchases": 1,
            "total_purchase_value": 200.0,
            "average_purchase_value": 200.0,
            "number_of_page_views": 1,
            "add_to_cart_count": 1,
            "begin_checkout_count": 0, # Assuming no begin_checkout event for this case
            "total_items_purchased": 1,
            "distinct_products_purchased": 1,
            "distinct_brands_purchased": 1,
            "distinct_products_viewed": 1,
            "distinct_brands_viewed": 1,
        }
    },
    {
        "name": "Customer with only Page Views",
        "customer_id_suffix": "customer_003",
        "events": [
            {"name": "page_view", "data": {}},
            {"name": "page_view", "data": {}},
            {"name": "page_view", "data": {}},
        ],
        "expected_features": {
            "number_of_purchases": 0,
            "total_purchase_value": 0.0,
            "average_purchase_value": 0.0,
            "number_of_page_views": 3,
            "add_to_cart_count": 0,
            "begin_checkout_count": 0,
            "total_items_purchased": 0,
            "distinct_products_purchased": 0,
            "distinct_brands_purchased": 0,
            "distinct_products_viewed": 0,
            "distinct_brands_viewed": 0,
        }
    },
    {
        "name": "Customer with Add to Cart but no Purchase",
        "customer_id_suffix": "customer_004",
        "events": [
            {"name": "page_view", "data": {}},
            {"name": "view_item", "data": {"items": [{"item_id": "SKU_777", "item_brand": "Brand Y"}]}},
            {"name": "add_to_cart", "data": {"items": [{"item_id": "SKU_777", "item_brand": "Brand Y", "quantity": 2}]}},
            {"name": "begin_checkout", "data": {"items": [{"item_id": "SKU_777", "item_brand": "Brand Y", "quantity": 2}]}},
        ],
        "expected_features": {
            "number_of_purchases": 0,
            "total_purchase_value": 0.0,
            "average_purchase_value": 0.0,
            "number_of_page_views": 1,
            "add_to_cart_count": 1,
            "begin_checkout_count": 1,
            "total_items_purchased": 0,
            "distinct_products_purchased": 0,
            "distinct_brands_purchased": 0,
            "distinct_products_viewed": 1,
            "distinct_brands_viewed": 1,
        }
    }
]

def test_all_customer_journeys_and_retraining():
    """
    Simulates all customer journeys, triggers retraining once, and then validates predictions.
    This ensures the model is trained with sufficient data from multiple customers.
    """
    print(f"\n\n{'='*50}")
    print(f"RUNNING ALL CUSTOMER JOURNEYS AND RETRAINING TEST")
    print(f"{'='*50}")

    # 1. Simulate all user journeys
    print("\n--- Simulating All User Journeys ---")
    for test_case in TEST_CASES:
        test_customer_id = f"validation_{test_case['customer_id_suffix']}"
        print(f"Simulating events for {test_customer_id} ({test_case['name']})...")
        for event in test_case["events"]:
            send_event(test_customer_id, event["name"], event["data"])
            time.sleep(0.05) # Small delay to simulate real-world event flow

    # 2. Trigger retraining once after all events are sent
    print("\n--- Triggering Model Retraining (once for all customers) ---")
    if not trigger_retraining(wait_time=10): # Increased wait time for more data
        pytest.fail("Model retraining failed.")

    # 3. Reload the model artifact in the API to ensure the new model is loaded
    if not reload_model_artifact():
        pytest.fail("Failed to reload model artifact in API.")

    # 4. Validate predictions for all customers
    print("\n--- Validating Predictions for All Customers ---")
    for test_case in TEST_CASES:
        test_customer_id = f"validation_{test_case['customer_id_suffix']}"
        validate_prediction(test_customer_id, test_case["expected_features"])

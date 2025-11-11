
import requests
import json
import time
import subprocess
import sys
import os

import json
import time
import sys
import os
from test_utils import (
    clear_database,
    send_event,
    get_prediction,
    trigger_retraining
)

def validate_prediction(customer_id, expected_features):
    """Gets a pLTV prediction and validates the returned features."""
    print(f"\n--- Validating Prediction for {customer_id} ---")
    
    response_data = get_prediction(customer_id)
    if not response_data:
        print("❌ FAILED: Did not receive a prediction response.")
        return

    features = response_data.get('features', {})
    pltv = response_data.get('pltv')

    print("\n--- Returned Data ---")
    print(f"pLTV: {pltv}")
    print("Features:", json.dumps(features, indent=2))
    
    print("\n--- Verification ---")
    validation_passed = True
    for key, expected_value in expected_features.items():
        # Use .get() to avoid KeyError for features that might not be created
        # (e.g., purchase features for a user with no purchases)
        actual_value = features.get(key)
        
        # Special handling for float comparison
        if isinstance(expected_value, float):
            if actual_value is None or not isinstance(actual_value, (int, float)) or abs(actual_value - expected_value) > 1e-6:
                validation_passed = False
                print(f"❌ FAILED: Feature '{key}'. Expected: {expected_value}, Got: {actual_value}")
            else:
                print(f"✅ PASSED: Feature '{key}'. Expected: {expected_value}, Got: {actual_value:.2f}")
        else:
            if actual_value != expected_value:
                validation_passed = False
                print(f"❌ FAILED: Feature '{key}'. Expected: {expected_value}, Got: {actual_value}")
            else:
                print(f"✅ PASSED: Feature '{key}'. Expected: {expected_value}, Got: {actual_value}")

    # The model is complex, so we don't expect an exact value, just a plausible one.
    if pltv is not None and pltv > 0:
        print(f"✅ PASSED: pLTV value is positive. Got: {pltv}")
    else:
        validation_passed = False
        print(f"❌ FAILED: pLTV value should be positive. Got: {pltv}")

    print("\n-------------------------")
    if validation_passed:
        print("✅ ✅ ✅ VALIDATION SUCCEEDED ✅ ✅ ✅")
    else:
        print("❌ ❌ ❌ VALIDATION FAILED ❌ ❌ ❌")
        # sys.exit(1) # Optional: uncomment to make test failure stop a CI/CD pipeline
    print("-------------------------")



# --- Test Case Definition ---

TEST_CASES = [
    {
        "name": "Full Journey with Two Purchases",
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
    # --- Add more test cases here in the future ---
]

def trigger_retraining():
    """Calls the /retrain endpoint and waits for it to likely complete."""
    print("\n--- Triggering Model Retraining ---")
    retrain_secret_key = os.environ.get("RETRAIN_SECRET_KEY", "YOUR_SECRET_KEY")
    if retrain_secret_key == "YOUR_SECRET_KEY":
        print("SKIPPING: RETRAIN_SECRET_KEY is not set.")
        return False
        
    url = f"{API_BASE_URL}/retrain?secret={retrain_secret_key}"
    try:
        response = requests.post(url)
        response.raise_for_status()
        print("Retraining triggered successfully.")
        # Wait for the background process to likely finish
        print("Waiting 5 seconds for retraining to complete...")
        time.sleep(5)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error triggering retraining: {e}")
        return False

def main():
    """Main function to run the validation test suite."""
    try:
        for i, case in enumerate(TEST_CASES):
            print(f"\n\n{'='*50}")
            print(f"RUNNING TEST CASE {i+1}: {case['name']}")
            print(f"{'='*50}")
            
            test_customer_id = f"validation_customer_{i+1:03d}"

            # 1. Clear the database before each test case
            clear_database()

            # 2. Simulate the user journey
            print(f"\n--- Simulating User Journey for {test_customer_id} ---")
            for event in case["events"]:
                send_event(test_customer_id, event["name"], event["data"])
                time.sleep(0.1)

            # 3. Trigger retraining to ensure a model exists for prediction
            if not trigger_retraining(wait_time=5):
                print("Skipping prediction validation as retraining was not triggered.")
                continue

            # 4. Get and validate the prediction and features
            validate_prediction(test_customer_id, case["expected_features"])
            
    except RuntimeError as e:
        print(f"\n\n❌ A critical error occurred during the test run: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


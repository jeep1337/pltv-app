import pytest
import requests
import json
import time
import os
import sys

from test_utils import (
    clear_database,
    send_event,
    get_prediction,
    trigger_retraining,
    reload_model_artifact
)

# --- Configuration ---
API_BASE_URL = os.environ.get("PLTV_API_BASE_URL", "http://127.0.0.1:5000")
RETRAIN_SECRET_KEY = os.environ.get("RETRAIN_SECRET_KEY", "YOUR_SECRET_KEY") 

# --- Test Logic ---

def test_model_retraining():
    """Runs the full test sequence for model retraining and reloading."""

    TEST_CUSTOMER_ID = "retrain_test_customer_001"
    TEST_CUSTOMER_ID_2 = "retrain_test_customer_002" # Add a second customer for training data
    TEST_CUSTOMER_ID_3 = "retrain_test_customer_003"
    TEST_CUSTOMER_ID_4 = "retrain_test_customer_004"

    # 1. Initial Setup
    clear_database()
    
    # 2. Send first events to seed initial data for two customers
    print("\n--- Step 1: Seeding initial data for four customers ---")
    send_event(TEST_CUSTOMER_ID, "purchase", {"value": 100.0})
    send_event(TEST_CUSTOMER_ID_2, "purchase", {"value": 50.0}) # Second customer
    send_event(TEST_CUSTOMER_ID_3, "purchase", {"value": 120.0})
    send_event(TEST_CUSTOMER_ID_4, "purchase", {"value": 80.0})
    time.sleep(0.5) # Give some time for events to be processed

    # 3. Trigger initial retraining
    print("\n--- Step 2: Triggering initial retraining ---")
    if not trigger_retraining(wait_time=5):
        pytest.fail("Initial retraining failed.")
    
    # 4. Reload the model artifact in the API
    if not reload_model_artifact():
        pytest.fail("Failed to reload model artifact after initial retraining.")

    # 5. Get baseline prediction for TEST_CUSTOMER_ID
    print("\n--- Step 3: Getting baseline prediction ---")
    pred_1 = get_prediction(TEST_CUSTOMER_ID)
    if pred_1 is None:
        pytest.fail("Baseline prediction failed.")

    # 6. Send a second event for TEST_CUSTOMER_ID
    print("\n--- Step 4: Sending second event to update customer history ---")
    send_event(TEST_CUSTOMER_ID, "purchase", {"value": 250.0})
    time.sleep(0.5) # Give some time for event to be processed
    
    # 7. Get prediction BEFORE retraining (should use the OLD model)
    print("\n--- Step 5: Getting prediction BEFORE retraining ---")
    print("(This should use the OLD model loaded in memory)")
    pred_2 = get_prediction(TEST_CUSTOMER_ID)
    if pred_2 is None:
        pytest.fail("Prediction before retraining failed.")

    # 8. Trigger retraining to load the new model
    print("\n--- Step 6: Triggering retraining to load new model ---")
    if not trigger_retraining(wait_time=5):
        pytest.fail("Retraining after second event failed.")
    
    # 9. Reload the model artifact in the API
    if not reload_model_artifact():
        pytest.fail("Failed to reload model artifact after second retraining.")

    # 10. Get prediction AFTER retraining (should use the NEWLY reloaded model)
    print("\n--- Step 7: Getting prediction AFTER retraining ---")
    print("(This should use the NEWLY reloaded model)")
    pred_3 = get_prediction(TEST_CUSTOMER_ID)
    if pred_3 is None:
        pytest.fail("Prediction after retraining failed.")

    # 11. Verification
    print("\n--- Step 8: Verifying results ---")

    print(f"Prediction 1 (Baseline):      Got={pred_1}")
    print(f"Prediction 2 (Pre-Retraining):  Got={pred_2}")
    print(f"Prediction 3 (Post-Retraining): Got={pred_3}")

    # --- Verification Logic ---
    
    # Baseline prediction should be positive
    assert pred_1 > 0, f"Baseline prediction ({pred_1}) should be a positive number."

    # Prediction before retraining should be the same as baseline (old model)
    assert pred_1 == pred_2, f"Prediction before retraining ({pred_2}) should be the same as baseline ({pred_1})."

    # Prediction after retraining should be different from the pre-retraining prediction (new model)
    assert pred_2 != pred_3, f"Prediction after retraining ({pred_3}) should be different from the pre-retraining prediction ({pred_2})."

    print("\n✅ ✅ ✅ TEST PASSED: Model reloading works as expected! ✅ ✅ ✅")

import requests
import json
import time
import subprocess
import sys
import os

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:5000"
# IMPORTANT: You must set this environment variable to match your secret key
RETRAIN_SECRET_KEY = os.environ.get("RETRAIN_SECRET_KEY", "YOUR_SECRET_KEY") 
TEST_CUSTOMER_ID = "retrain_test_customer_001"

from test_utils import (
    clear_database,
    send_event,
    get_prediction,
    trigger_retraining
)

# --- Main Test Logic ---

def main():

    """Runs the full test sequence for model retraining and reloading."""

    try:

        TEST_CUSTOMER_ID = "retrain_test_customer_001"



        # 1. Initial Setup

        clear_database()

        

        # 2. Send first event and train the initial model

        print("\n--- Step 1: Seeding initial data and training first model ---")

        send_event(TEST_CUSTOMER_ID, "purchase", value=100.0)

        if not trigger_retraining(wait_time=5):

            sys.exit(1)



        # 3. Get baseline prediction

        print("\n--- Step 2: Getting baseline prediction ---")

        pred_1_response = get_prediction(TEST_CUSTOMER_ID)

        pred_1 = pred_1_response.get('pltv') if pred_1_response else None



        # 4. Send a second event

        print("\n--- Step 3: Sending second event to update customer history ---")

        send_event(TEST_CUSTOMER_ID, "purchase", value=250.0)

        

        # 5. Get prediction BEFORE retraining

        print("\n--- Step 4: Getting prediction BEFORE retraining ---")

        print("(This should use the OLD model loaded in memory)")

        pred_2_response = get_prediction(TEST_CUSTOMER_ID)

        pred_2 = pred_2_response.get('pltv') if pred_2_response else None



        # 6. Trigger retraining to load the new model

        print("\n--- Step 5: Triggering retraining to load new model ---")

        if not trigger_retraining(wait_time=5):

            sys.exit(1)



        # 7. Get prediction AFTER retraining

        print("\n--- Step 6: Getting prediction AFTER retraining ---")

        print("(This should use the NEWLY reloaded model)")

        pred_3_response = get_prediction(TEST_CUSTOMER_ID)

        pred_3 = pred_3_response.get('pltv') if pred_3_response else None



        # 8. Verification

        print("\n--- Step 7: Verifying results ---")



        print(f"Prediction 1 (Baseline):      Got={pred_1}")

        print(f"Prediction 2 (Pre-Retraining):  Got={pred_2}")

        print(f"Prediction 3 (Post-Retraining): Got={pred_3}")



        # --- Verification Logic ---

        test_passed = True

        

        if pred_1 is None or pred_1 <= 0:

            print(f"\n❌ FAILED: Baseline prediction ({pred_1}) should be a positive number.")

            test_passed = False

        else:

            print(f"\n✅ PASSED: Baseline prediction ({pred_1}) is a positive number.")



        if pred_1 != pred_2:

            print(f"\n❌ FAILED: Prediction before retraining ({pred_2}) should be the same as baseline ({pred_1}).")

            test_passed = False

        else:

            print(f"\n✅ PASSED: Prediction before retraining is correctly unchanged.")



        if pred_2 == pred_3:

            print(f"\n❌ FAILED: Prediction after retraining ({pred_3}) should be different from the pre-retraining prediction ({pred_2}).")

            test_passed = False

        else:

            print(f"\n✅ PASSED: Prediction after retraining has correctly changed.")



        if test_passed:

            print("\n✅ ✅ ✅ TEST PASSED: Model reloading works as expected! ✅ ✅ ✅")

        else:

            print("\n❌ ❌ ❌ TEST FAILED: Model was not reloaded correctly. ❌ ❌ ❌")

            sys.exit(1)



    except RuntimeError as e:

        print(f"\n\n❌ A critical error occurred during the test run: {e}", file=sys.stderr)

        sys.exit(1)




if __name__ == "__main__":
    main()

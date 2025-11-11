
import json
import os
import sys
from database import db
from dotenv import load_dotenv

def main():
    """Retrieves all customer events and prints them."""
    print("--- Retrieving all customer events ---")
    
    try:
        # Load environment variables to ensure database connection is available
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
        
        if not os.environ.get("DATABASE_URL"):
            print("Error: DATABASE_URL not set. Please create a .env file or set the environment variable.", file=sys.stderr)
            sys.exit(1)

        all_events = db.get_all_customer_events()

        if not all_events:
            print("No events found in the database.")
            return

        for customer_id, event_data, created_at in all_events:
            print(f"\n{'='*20}")
            print(f"Customer ID: {customer_id}")
            print(f"Created At: {created_at}")
            print("Event Data:")
            # The data is a list of JSON objects, where each object has an 'events' key
            # We should iterate and print them cleanly.
            if isinstance(event_data, list):
                for record in event_data:
                    print(json.dumps(record, indent=2))
            else: # Fallback for the older format
                 print(json.dumps(event_data, indent=2))
            print(f"{'='*20}")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


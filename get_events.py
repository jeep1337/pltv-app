
import json
from database import get_all_customer_events

def main():
    """Retrieves all customer events and prints them."""
    print("--- Retrieving all customer events ---")
    all_events = get_all_customer_events()

    if not all_events:
        print("No events found in the database.")
        return

    for customer_id, event_data, created_at in all_events:
        print(f"Customer ID: {customer_id}")
        print(f"Created At: {created_at}")
        print("Event Data:")
        print(json.dumps(event_data, indent=2))
        print("--------------------")

if __name__ == "__main__":
    main()

from database import db
import os
import sys
import argparse
from urllib.parse import urlparse

def clear_all_data(force=False):
    """
    Clears all data from both customers and customer_features tables.
    Includes a safety confirmation prompt unless --force is used.
    """
    print("--- Database Reset Script ---")
    
    try:
        # Load environment variables to get the database URL
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
        
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("Error: DATABASE_URL environment variable not set.", file=sys.stderr)
            sys.exit(1)
            
        parsed_url = urlparse(db_url)
        db_info = f"Host: {parsed_url.hostname}, DB: {parsed_url.path.lstrip('/')}"

        if not force:
            print("\nWARNING: This is a destructive operation that will delete all data from:")
            print(f"  - customers")
            print(f"  - customer_features")
            print(f"\nOn database: {db_info}\n")
            
            confirm = input("Are you sure you want to continue? Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Operation cancelled.")
                return

        print("\nClearing all tables...")
        db.clear_all_tables()
        print("--- All tables cleared successfully ---")

    except Exception as e:
        print(f"An error occurred while clearing tables: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear all data from the database.")
    parser.add_argument('-f', '--force', action='store_true', help="Force deletion without confirmation prompt.")
    args = parser.parse_args()
    
    clear_all_data(force=args.force)



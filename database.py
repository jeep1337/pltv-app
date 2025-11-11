import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

class Database:
    def __init__(self, min_conn=1, max_conn=10):
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        self.pool = pool.SimpleConnectionPool(min_conn, max_conn, self.database_url)

    @contextmanager
    def get_connection(self):
        """Context manager to get a connection from the pool."""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit=False):
        """Context manager to get a cursor from a connection."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()

    def create_all_tables(self):
        """Creates all necessary tables if they don't exist."""
        with self.get_cursor(commit=True) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id VARCHAR(255) PRIMARY KEY,
                    event_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customer_features (
                    id SERIAL PRIMARY KEY,
                    customer_id VARCHAR(255) UNIQUE NOT NULL,
                    total_purchase_value FLOAT DEFAULT 0,
                    number_of_purchases INTEGER DEFAULT 0,
                    average_purchase_value FLOAT DEFAULT 0,
                    total_items_purchased INTEGER DEFAULT 0,
                    distinct_products_purchased INTEGER DEFAULT 0,
                    distinct_brands_purchased INTEGER DEFAULT 0,
                    distinct_products_viewed INTEGER DEFAULT 0,
                    distinct_brands_viewed INTEGER DEFAULT 0,
                    number_of_page_views INTEGER DEFAULT 0,
                    days_since_last_purchase INTEGER DEFAULT 0,
                    time_since_first_event INTEGER DEFAULT 0,
                    purchase_frequency FLOAT DEFAULT 0,
                    pltv FLOAT DEFAULT 0,
                    add_to_cart_count INTEGER DEFAULT 0,
                    begin_checkout_count INTEGER DEFAULT 0,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
        print("All tables created or already exist.")

    def get_all_customer_events(self):
        """Retrieves all customer event data from the database."""
        with self.get_cursor() as cur:
            cur.execute("SELECT customer_id, event_data, created_at FROM customers")
            return cur.fetchall()

    def clear_customers_table(self):
        """Clears the customers table."""
        with self.get_cursor(commit=True) as cur:
            cur.execute("DELETE FROM customers")
        print("Customers table cleared.")

    def get_customer_events(self, customer_id):
        """Retrieves all event data for a specific customer."""
        with self.get_cursor() as cur:
            cur.execute("SELECT customer_id, event_data, created_at FROM customers WHERE customer_id = %s", (customer_id,))
            return cur.fetchall()

    def get_customer_features(self, customer_id):
        """Retrieves pre-aggregated features for a specific customer."""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM customer_features WHERE customer_id = %s", (customer_id,))
            features = cur.fetchone()
            if features:
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, features))
            return None

    def clear_customer_features_table(self):
        """Clears the customer_features table."""
        with self.get_cursor(commit=True) as cur:
            cur.execute("DELETE FROM customer_features")
        print("Customer features table cleared.")

    def clear_all_tables(self):
        """Clears all data from all tables for a clean slate."""
        self.clear_customers_table()
        self.clear_customer_features_table()

    def upsert_customer_features(self, features_dict):
        """Inserts or updates a customer's features in the database securely."""
        # Whitelist of allowed columns to prevent SQL injection on column names
        allowed_columns = {
            'customer_id', 'total_purchase_value', 'number_of_purchases', 'average_purchase_value',
            'total_items_purchased', 'distinct_products_purchased', 'distinct_brands_purchased',
            'distinct_products_viewed', 'distinct_brands_viewed', 'number_of_page_views',
            'days_since_last_purchase', 'time_since_first_event', 'purchase_frequency', 'pltv',
            'add_to_cart_count', 'begin_checkout_count'
        }
        
        # Filter the dictionary to only include allowed columns
        filtered_features = {k: v for k, v in features_dict.items() if k in allowed_columns}
        
        if 'customer_id' not in filtered_features:
            raise ValueError("customer_id is a required key for upserting features.")

        columns = ", ".join(filtered_features.keys())
        placeholders = ", ".join([f"%({key})s" for key in filtered_features.keys()])
        update_str = ", ".join([f"{key} = EXCLUDED.{key}" for key in filtered_features if key != 'customer_id'])

        query = f"""
            INSERT INTO customer_features ({columns}) 
            VALUES ({placeholders}) 
            ON CONFLICT (customer_id) 
            DO UPDATE SET {update_str}, updated_at = CURRENT_TIMESTAMP
        """
        
        with self.get_cursor(commit=True) as cur:
            cur.execute(query, filtered_features)

    def update_features_incrementally(self, customer_id, event):
        """
        Incrementally and atomically updates a customer's features based on a single event
        using a single UPSERT query. This is far more efficient than a full recalculation.
        """
        event_name = event.get('event_name') or event.get('event_type')
        if not event_name:
            return

        query = None
        params = {'customer_id': customer_id}

        if event_name == 'page_view':
            query = """
                INSERT INTO customer_features (customer_id, number_of_page_views) VALUES (%(customer_id)s, 1)
                ON CONFLICT (customer_id) DO UPDATE SET
                    number_of_page_views = customer_features.number_of_page_views + 1,
                    updated_at = CURRENT_TIMESTAMP;
            """
        elif event_name == 'add_to_cart':
            query = """
                INSERT INTO customer_features (customer_id, add_to_cart_count) VALUES (%(customer_id)s, 1)
                ON CONFLICT (customer_id) DO UPDATE SET
                    add_to_cart_count = customer_features.add_to_cart_count + 1,
                    updated_at = CURRENT_TIMESTAMP;
            """
        elif event_name == 'begin_checkout':
            query = """
                INSERT INTO customer_features (customer_id, begin_checkout_count) VALUES (%(customer_id)s, 1)
                ON CONFLICT (customer_id) DO UPDATE SET
                    begin_checkout_count = customer_features.begin_checkout_count + 1,
                    updated_at = CURRENT_TIMESTAMP;
            """
        elif event_name == 'purchase':
            purchase_value = float(event.get('value', 0.0))
            params['purchase_value'] = purchase_value
            query = """
                INSERT INTO customer_features (customer_id, number_of_purchases, total_purchase_value, days_since_last_purchase)
                VALUES (%(customer_id)s, 1, %(purchase_value)s, 0)
                ON CONFLICT (customer_id) DO UPDATE SET
                    number_of_purchases = customer_features.number_of_purchases + 1,
                    total_purchase_value = customer_features.total_purchase_value + %(purchase_value)s,
                    days_since_last_purchase = 0,
                    updated_at = CURRENT_TIMESTAMP;
            """
        
        if query:
            with self.get_cursor(commit=True) as cur:
                cur.execute(query, params)
        
        # Note: This incremental update doesn't update complex, dependent features like
        # average_purchase_value or purchase_frequency. A full recalculation is still
        # needed for those. A hybrid approach (e.g., full recalc only on purchase) is the next step.


# --- Global Database Instance ---
# This instance will be imported by other parts of the application
db = Database()

if __name__ == "__main__":
    print("Initializing database and creating tables...")
    db.create_all_tables()
    print("Database setup complete.")


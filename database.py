import os
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://neondb_owner:npg_dmSxUj15yGoI@ep-shy-leaf-ag899uvw-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

def connect_db():
    """Establishes a connection to the Neon database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("Database connection established successfully.")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def create_customers_table():
    """Creates a customers table if it doesn't exist."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id VARCHAR(255) PRIMARY KEY,
                    event_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            print("Customers table created or already exists.")
        except Exception as e:
            print(f"Error creating customers table: {e}")
        finally:
            conn.close()

def clear_customers_table():
    """Clears the customers table."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM customers")
            conn.commit()
            print("Customers table cleared.")
        except Exception as e:
            print(f"Error clearing customers table: {e}")
        finally:
            conn.close()

def get_all_customer_events():
    """Retrieves all customer event data from the database."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT customer_id, event_data, created_at FROM customers")
            events = cur.fetchall()
            return events
        except Exception as e:
            print(f"Error retrieving customer events: {e}")
            return []
        finally:
            conn.close()
    return []

def get_customer_events(customer_id):
    """Retrieves all event data for a specific customer from the database."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT customer_id, event_data, created_at FROM customers WHERE customer_id = %s", (customer_id,))
            events = cur.fetchall()
            return events
        except Exception as e:
            print(f"Error retrieving events for customer {customer_id}: {e}")
            return []
        finally:
            conn.close()


def create_customer_features_table():
    """Creates a customer_features table if it doesn't exist."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
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
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            print("Customer features table created or already exists.")
        except Exception as e:
            print(f"Error creating customer_features table: {e}")
        finally:
            conn.close()

def get_customer_features(customer_id):
    """Retrieves pre-aggregated features for a specific customer."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM customer_features WHERE customer_id = %s", (customer_id,))
            features = cur.fetchone()
            if features:
                # Convert tuple to dict
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, features))
            return None
        except Exception as e:
            print(f"Error retrieving features for customer {customer_id}: {e}")
            return None
        finally:
            conn.close()
    return None

def upsert_customer_features(features_dict):
    """Inserts or updates a customer's features in the database."""
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            
            columns = ", ".join(features_dict.keys())
            placeholders = ", ".join([f"%({key})s" for key in features_dict.keys()])
            
            update_str = ", ".join([f"{key} = EXCLUDED.{key}" for key in features_dict.keys() if key != 'customer_id'])
            
            query = f"""
                INSERT INTO customer_features ({columns}) 
                VALUES ({placeholders}) 
                ON CONFLICT (customer_id) 
                DO UPDATE SET {update_str}, updated_at = CURRENT_TIMESTAMP
            """
            
            cur.execute(query, features_dict)
            conn.commit()
        except Exception as e:
            print(f"Error upserting customer features: {e}")
            conn.rollback()
        finally:
            conn.close()

if __name__ == "__main__":
    create_customers_table()
    create_customer_features_table()
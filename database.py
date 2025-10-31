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

if __name__ == "__main__":
    create_customers_table()
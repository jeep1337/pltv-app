from flask import Flask, request, jsonify
from database import connect_db, create_customers_table, get_all_customer_events
import json
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('pltv_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get customer data from the request
    customer_data = request.get_json()

    # Preprocess the input data (this should match the preprocessing in model.py)
    total_purchase_value = customer_data.get('total_purchase_value', 0)
    number_of_purchases = customer_data.get('number_of_purchases', 0)
    number_of_page_views = customer_data.get('number_of_page_views', 0)
    days_since_last_purchase = customer_data.get('days_since_last_purchase', 0)
    prediction_data = [[total_purchase_value, number_of_purchases, number_of_page_views, days_since_last_purchase]]

    # Use the model to predict pLTV
    pltv = model.predict(prediction_data)[0]

    # Return the prediction
    return jsonify({'pltv': pltv})

@app.route('/event', methods=['POST'])
def event():
    event_data = request.get_json()
    customer_id = event_data.get('customer_id')
    event_payload = event_data.get('event_data')

    if not customer_id or not event_payload:
        return jsonify({'error': 'customer_id and event_data are required'}), 400

    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            print(f"Received event for customer_id: {customer_id}")
            print(f"Event payload: {event_payload}")
            cur.execute(
                "INSERT INTO customers (customer_id, event_data) VALUES (%s, %s)",
                (customer_id, json.dumps(event_payload))
            )
            conn.commit()
            print("Event data committed to database.")
            return jsonify({'message': 'Event data stored successfully'}), 200
        except Exception as e:
            print(f"Error storing event data: {e}")
            conn.rollback() # Rollback in case of error
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
    print("Database connection failed in event() function.")
    return jsonify({'error': 'Database connection failed'}), 500

if __name__ == '__main__':
    create_customers_table()
    app.run(debug=True)
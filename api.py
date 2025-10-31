import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from database import connect_db, create_customers_table, get_all_customer_events, get_customer_events
from model import preprocess_data

app = Flask(__name__)

# Load the trained model
model = joblib.load('pltv_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get customer_id from the request
    data = request.get_json()
    customer_id = data.get('customer_id')

    if not customer_id:
        return jsonify({'error': 'customer_id is required'}), 400

    # Retrieve all events for the customer
    events_raw = get_customer_events(customer_id)

    if not events_raw:
        return jsonify({'error': f'No events found for customer_id: {customer_id}'}), 404

    # Convert raw events to a DataFrame suitable for preprocess_data
    # preprocess_data expects a DataFrame with 'customer_id', 'event_data', 'created_at'
    df = pd.DataFrame(events_raw, columns=['customer_id', 'event_data', 'created_at'])

    # Preprocess the data to generate features
    # preprocess_data returns a DataFrame with all customer features
    customer_features_df = preprocess_data(df)

    # Filter for the specific customer's features
    customer_features = customer_features_df[customer_features_df['customer_id'] == customer_id]

    if customer_features.empty:
        return jsonify({'error': f'Could not generate features for customer_id: {customer_id}'}), 500

    # Ensure the order of features matches the training data
    # X = df[['total_purchase_value', 'number_of_purchases', 'number_of_page_views', 'days_since_last_purchase', 'purchase_frequency', 'average_purchase_value']]
    prediction_data = customer_features[[
        'total_purchase_value',
        'number_of_purchases',
        'number_of_page_views',
        'days_since_last_purchase',
        'purchase_frequency',
        'average_purchase_value'
    ]].values

    # Use the model to predict pLTV
    pltv = model.predict(prediction_data)[0]

    # Return the prediction
    return jsonify({'customer_id': customer_id, 'pltv': pltv})

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

            # Check if customer_id already exists
            cur.execute("SELECT event_data FROM customers WHERE customer_id = %s", (customer_id,))
            existing_record = cur.fetchone()

            if existing_record:
                # Customer exists, update event_data by appending new event
                existing_event_data = existing_record[0] # This is already a Python dict from JSONB
                if 'events' not in existing_event_data or not isinstance(existing_event_data['events'], list):
                    existing_event_data['events'] = [] # Initialize if not an array

                # event_payload from request is {'events': [...]}
                # We need to append the *contents* of event_payload['events']
                if 'events' in event_payload and isinstance(event_payload['events'], list):
                    existing_event_data['events'].extend(event_payload['events'])

                cur.execute(
                    "UPDATE customers SET event_data = %s, created_at = CURRENT_TIMESTAMP WHERE customer_id = %s",
                    (json.dumps(existing_event_data), customer_id)
                )
                print(f"Event data updated for customer_id: {customer_id}")
            else:
                # Customer does not exist, insert new record
                cur.execute(
                    "INSERT INTO customers (customer_id, event_data) VALUES (%s, %s)",
                    (customer_id, json.dumps(event_payload))
                )
                print(f"New event data inserted for customer_id: {customer_id}")
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
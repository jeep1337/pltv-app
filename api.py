import json
import time
import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify
from database import connect_db, create_customers_table, get_all_customer_events, get_customer_events
from model import preprocess_data

app = Flask(__name__)

# Construct path to the model file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pltv_model.pkl')

# Load the trained model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Prediction request received')
    # Get customer_id from the request
    data = request.get_json()
    customer_id = data.get('customer_id')
    app.logger.info(f'customer_id: {customer_id}')

    if not customer_id:
        return jsonify({'error': 'customer_id is required'}), 400

    # Retrieve all events for the customer
    app.logger.info('Retrieving customer events')
    events_raw = get_customer_events(customer_id)
    app.logger.info(f'events_raw: {events_raw}')

    if not events_raw:
        return jsonify({'error': f'No events found for customer_id: {customer_id}'}), 404

    # Convert raw events to a DataFrame suitable for preprocess_data
    app.logger.info('Converting to DataFrame')
    df = pd.DataFrame(events_raw, columns=['customer_id', 'event_data', 'created_at'])
    app.logger.info(f'DataFrame created: {df.head()}')

    # Preprocess the data to generate features
    app.logger.info('Preprocessing data')
    customer_features_df = preprocess_data(df)
    app.logger.info(f'Preprocessing complete: {customer_features_df.head()}')

    # Filter for the specific customer's features
    app.logger.info('Filtering for customer features')
    customer_features = customer_features_df[customer_features_df['customer_id'] == customer_id]
    app.logger.info(f'Customer features: {customer_features}')

    if customer_features.empty:
        return jsonify({'error': f'Could not generate features for customer_id: {customer_id}'}), 500

    # Ensure the order of features matches the training data
    app.logger.info('Preparing prediction data')
    prediction_data = customer_features[[
        'total_purchase_value',
        'number_of_purchases',
        'number_of_page_views',
        'days_since_last_purchase',
        'purchase_frequency',
        'average_purchase_value'
    ]].values
    app.logger.info(f'Prediction data: {prediction_data}')

    feature_names = [
        'total_purchase_value',
        'number_of_purchases',
        'number_of_page_views',
        'days_since_last_purchase',
        'purchase_frequency',
        'average_purchase_value'
    ]
    prediction_data_df = pd.DataFrame(prediction_data, columns=feature_names)
    app.logger.info(f'Prediction data DataFrame: {prediction_data_df}')

    # Use the model to predict pLTV
    app.logger.info('Predicting pLTV')
    pltv = model.predict(prediction_data_df)[0]
    app.logger.info(f'pLTV predicted: {pltv}')

    # Convert features to a dictionary for the JSON response
    features_dict = customer_features.to_dict('records')[0]

    # Return the prediction and the features
    return jsonify({
        'customer_id': customer_id,
        'pltv': pltv,
        'features': features_dict
    })

@app.route('/event', methods=['POST'])
def event():
    event_data = request.get_json()
    customer_id = event_data.get('customer_id')
    event_payload = event_data.get('event_data')

    if not customer_id or not event_payload:
        return jsonify({'error': 'customer_id and event_data are required'}), 400

    # --- Start: Normalize event payload and Inject Timestamp ---
    # This makes the endpoint more robust to different sGTM data structures.
    if isinstance(event_payload, dict) and 'events' not in event_payload:
        # If payload is a single event, wrap it in the expected list structure.
        event_payload = {'events': [event_payload]}

    if 'events' in event_payload and isinstance(event_payload['events'], list):
        current_timestamp_micros = int(time.time() * 1_000_000)
        for event in event_payload['events']:
            if isinstance(event, dict):
                event['api_timestamp_micros'] = current_timestamp_micros
    # --- End: Normalize event payload and Inject Timestamp ---

    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            print(f"Received event for customer_id: {customer_id}")
            print(f"Normalized Event payload (with server timestamp): {event_payload}")

            # Check if customer_id already exists
            cur.execute("SELECT event_data FROM customers WHERE customer_id = %s", (customer_id,))
            existing_record = cur.fetchone()

            if existing_record:
                # Customer exists, update event_data by appending new event
                existing_event_data = existing_record[0]
                if 'events' not in existing_event_data or not isinstance(existing_event_data['events'], list):
                    existing_event_data['events'] = []

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
            conn.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
    print("Database connection failed in event() function.")
    return jsonify({'error': 'Database connection failed'}), 500

if __name__ == '__main__':
    create_customers_table()
    app.run(debug=True)
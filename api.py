
import json
import time
import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify
from database import (
    connect_db, 
    create_customers_table, 
    get_customer_events,
    create_customer_features_table,
    get_customer_features,
    upsert_customer_features
)
from features import calculate_features

app = Flask(__name__)

# Construct path to the model file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pltv_model.pkl')

# Load the trained model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Prediction request received')
    data = request.get_json()
    customer_id = data.get('customer_id')
    app.logger.info(f'customer_id: {customer_id}')

    if not customer_id:
        return jsonify({'error': 'customer_id is required'}), 400

    # Retrieve pre-aggregated features
    app.logger.info('Retrieving customer features')
    features_dict = get_customer_features(customer_id)

    if not features_dict:
        return jsonify({'error': f'No features found for customer_id: {customer_id}'}), 404

    # Prepare features for prediction
    app.logger.info(f'Features retrieved: {features_dict}')
    customer_features = pd.DataFrame([features_dict])

    feature_names = [
        'total_purchase_value', 'number_of_purchases', 'number_of_page_views',
        'days_since_last_purchase', 'purchase_frequency', 'average_purchase_value',
        'total_items_purchased', 'distinct_products_purchased', 'distinct_brands_purchased',
        'distinct_products_viewed', 'distinct_brands_viewed'
    ]

    # Ensure all feature columns are present, fill with 0 if not
    for col in feature_names:
        if col not in customer_features.columns:
            customer_features[col] = 0
            
    prediction_data = customer_features[feature_names].values
    app.logger.info(f'Prediction data: {prediction_data}')

    # Predict pLTV
    pltv = model.predict(prediction_data)[0]
    app.logger.info(f'pLTV predicted: {pltv}')

    # Return prediction and features
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

    # --- Normalize and Timestamp Event ---
    if isinstance(event_payload, dict) and 'events' not in event_payload:
        event_payload = {'events': [event_payload]}

    if 'events' in event_payload and isinstance(event_payload['events'], list):
        current_timestamp_micros = int(time.time() * 1_000_000)
        for event in event_payload['events']:
            if isinstance(event, dict):
                event['api_timestamp_micros'] = current_timestamp_micros

    # --- Store Raw Event and Update Features ---
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # 1. Store the raw event
            cur.execute("SELECT event_data FROM customers WHERE customer_id = %s", (customer_id,))
            existing_record = cur.fetchone()
            if existing_record:
                existing_event_data = existing_record[0]
                if 'events' not in existing_event_data or not isinstance(existing_event_data['events'], list):
                    existing_event_data['events'] = []
                if 'events' in event_payload and isinstance(event_payload['events'], list):
                    existing_event_data['events'].extend(event_payload['events'])
                cur.execute(
                    "UPDATE customers SET event_data = %s, created_at = CURRENT_TIMESTAMP WHERE customer_id = %s",
                    (json.dumps(existing_event_data), customer_id)
                )
            else:
                cur.execute(
                    "INSERT INTO customers (customer_id, event_data) VALUES (%s, %s)",
                    (customer_id, json.dumps(event_payload))
                )
            conn.commit()

            # 2. Recalculate and upsert features
            all_events_raw = get_customer_events(customer_id)
            if all_events_raw:
                # Unpack all historical events into a single list
                all_individual_events = []
                for _, event_json, _ in all_events_raw:
                    if isinstance(event_json, dict) and 'events' in event_json and isinstance(event_json['events'], list):
                        for e in event_json['events']:
                            event_with_id = e.copy()
                            event_with_id['customer_id'] = customer_id
                            all_individual_events.append(event_with_id)
                
                events_df = pd.DataFrame(all_individual_events)
                
                # Calculate features
                customer_features_df = calculate_features(events_df)
                
                if not customer_features_df.empty:
                    # Convert DataFrame row to dictionary
                    features_dict = customer_features_df.iloc[0].to_dict()
                    # Upsert features
                    upsert_customer_features(features_dict)
                    app.logger.info(f"Features updated for customer_id: {customer_id}")

            return jsonify({'message': 'Event processed and features updated'}), 200

        except Exception as e:
            app.logger.error(f"Error processing event: {e}")
            conn.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
            
    return jsonify({'error': 'Database connection failed'}), 500

if __name__ == '__main__':
    create_customers_table()
    create_customer_features_table()
    app.run(debug=True)

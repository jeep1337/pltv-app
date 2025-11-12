
import json
import time
import joblib
import pandas as pd
import os
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from database import db  # Import the single db instance
from features import calculate_features
from model import retrain_and_save_model

# Construct path to the model file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pltv_model.pkl')

# Global variables for the loaded model and its features
model = None
model_features = []

def load_model_artifact():
    """Loads the model artifact from disk and populates global variables."""
    global model, model_features
    try:
        if os.path.exists(model_path):
            model_artifact = joblib.load(model_path)
            if not isinstance(model_artifact, dict) or 'model' not in model_artifact or 'features' not in model_artifact:
                app.logger.error("Model artifact 'pltv_model.pkl' is malformed or incomplete.")
                model = None
                model_features = []
                return
            model = model_artifact['model']
            model_features = model_artifact['features']
            app.logger.info(f"Model artifact loaded successfully. Features: {model_features}")
        else:
            app.logger.warning("Model artifact 'pltv_model.pkl' not found. Predictions will not be available until a model is trained.")
    except Exception as e:
        app.logger.error(f"Error loading model artifact: {e}")

app = Flask(__name__)

# Initialize the database and create tables if they don't exist
db.create_all_tables()
# Load the model artifact on startup
load_model_artifact()

@app.route('/event', methods=['PUT', 'POST'])
def event():
    try:
        app.logger.info(f"Incoming request Content-Type: {request.headers.get('Content-Type')}")
        app.logger.info(f"Incoming raw request data: {request.data.decode('utf-8', errors='ignore')}")
        event_data = request.get_json()
        if event_data is None:
            app.logger.error("Incoming request body is not valid JSON or is empty.")
            return jsonify({"error": "Invalid JSON or empty request body"}), 400
        app.logger.info(f"Incoming event data: {json.dumps(event_data, indent=2)}")

        # Check if the payload contains a list of events
        events = event_data.get('events', [event_data])
        if not isinstance(events, list):
            events = [events]

        for single_event in events:
            # --- Start: Customer ID Extraction Logic ---
            customer_id = single_event.get('user_pseudo_id') or single_event.get('client_id')

            if not customer_id and isinstance(single_event, dict):
                user_properties = single_event.get('user_properties')
                if user_properties and isinstance(user_properties, dict):
                    user_pseudo_id_obj = user_properties.get('user_pseudo_id')
                    if user_pseudo_id_obj and isinstance(user_pseudo_id_obj, dict):
                        customer_id = user_pseudo_id_obj.get('value')
                
                if not customer_id:
                    client_info = single_event.get('client_info')
                    if client_info and isinstance(client_info, dict):
                        customer_id = client_info.get('client_id')
                if not customer_id:
                    customer_id = single_event.get('_ga')

            if not customer_id:
                app.logger.error("Event data must contain 'user_pseudo_id' or 'client_id' at top level or in common nested structures.")
                # Continue to next event if one is malformed
                continue
            # --- End: Customer ID Extraction Logic ---

            db.upsert_event(customer_id, single_event)
            
            # --- Start: Hybrid Feature Update Logic ---
            event_name = single_event.get('event_name') or single_event.get('event_type')

            # For purchase events, trigger a full recalculation for the specific customer
            if event_name == 'purchase':
                try:
                    customer_events = db.get_customer_events(customer_id)
                    if customer_events:
                        features = calculate_features(customer_id, customer_events)
                        db.upsert_customer_features(features)
                        app.logger.info(f"Successfully recalculated and upserted features for customer {customer_id} after purchase.")
                except Exception as e:
                    app.logger.error(f"Error during full feature recalculation for customer {customer_id}: {e}")
            # For other common events, perform a fast, incremental update
            else:
                db.update_features_incrementally(customer_id, single_event)
            # --- End: Hybrid Feature Update Logic ---

        return jsonify({"message": "Events received and processed"}), 200
    except Exception as e:
        app.logger.error(f"Error processing event: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict', methods=['GET'])
def predict():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({"error": "customer_id is required"}), 400

    if model is None or not model_features:
        return jsonify({"error": "Model not loaded or trained yet. Please retrain the model."}), 503

    # Retrieve customer features from the database
    customer_features_dict = db.get_customer_features(customer_id)
    if not customer_features_dict:
        return jsonify({"error": f"No features found for customer_id: {customer_id}"}), 404

    # Convert features to DataFrame for prediction
    # Ensure the order of features matches the model's expected features
    features_df = pd.DataFrame([customer_features_dict])
    
    # Align columns with model_features, filling missing with 0
    X_predict = features_df[model_features].fillna(0)

    try:
        prediction = model.predict(X_predict)[0]
        return jsonify({"customer_id": customer_id, "predicted_pltv": prediction}), 200
    except Exception as e:
        app.logger.error(f"Error during prediction for customer {customer_id}: {e}")
        return jsonify({"error": "Error during prediction"}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    # Run retraining in a background thread to avoid blocking the API
    thread = threading.Thread(target=retrain_and_save_model)
    thread.daemon = True  # Allow the main program to exit even if the thread is still running
    thread.start()
    return jsonify({"message": "Model retraining initiated in the background."}), 202

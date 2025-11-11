
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

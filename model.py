
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from database import db
from features import calculate_features

def load_data():
    """Loads raw event data from the database."""
    events = db.get_all_customer_events()
    print("Raw events list from database:")
    # print(events) # This can be very verbose
    return events

def preprocess_data_for_training(raw_events):
    """
    Preprocesses the raw event data for model training in a memory-efficient way.
    It processes events per customer to avoid creating one massive DataFrame.
    """
    if not raw_events:
        return pd.DataFrame()

    # Group events by customer_id
    customer_events_map = {}
    for customer_id, event_data_dict, _ in raw_events:
        if customer_id not in customer_events_map:
            customer_events_map[customer_id] = []
        
        # event_data_dict is already a Python dictionary representing a single event
        event_with_id = event_data_dict.copy()
        if 'customer_id' not in event_with_id:
            event_with_id['customer_id'] = customer_id
        customer_events_map[customer_id].append(event_with_id)

    if not customer_events_map:
        return pd.DataFrame()

    # Calculate features for each customer individually and collect them
    all_customer_features = []
    for customer_id, events in customer_events_map.items():
        if not events:
            continue
        
        events_df = pd.DataFrame(events)
        customer_features_df = calculate_features(events_df)
        
        if not customer_features_df.empty:
            all_customer_features.append(customer_features_df)

    if not all_customer_features:
        return pd.DataFrame()

    # Concatenate all feature DataFrames into one
    final_features_df = pd.concat(all_customer_features, ignore_index=True)
    
    print("Customer Features DataFrame for Training (Memory-Efficient):")
    print(final_features_df)

    return final_features_df

def train_model(df):
    """
    Trains the RandomForestRegressor model, including hyperparameter tuning and validation.
    
    Returns:
        A dictionary containing the trained model, feature list, and performance metrics.
    """
    if df.empty or 'pltv' not in df.columns:
        return None

    # Dynamically determine feature columns, excluding identifiers and the target variable
    feature_columns = [col for col in df.columns if col not in ['customer_id', 'pltv']]
    
    X = df[feature_columns]
    y = df['pltv']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Hyperparameter Tuning with GridSearchCV ---
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, 
                               n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
    
    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # --- Evaluate the Best Model ---
    y_pred = best_model.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred))
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    
    print("\n--- Model Performance on Validation Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("-----------------------------------------\n")

    # --- Feature Importances ---
    importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Model Feature Importances:")
    print(importances)

    return {
        'model': best_model,
        'features': feature_columns,
        'metrics': {
            'mae': mae,
            'rmse': rmse
        }
    }

def save_model(model_artifact):
    """Saves the model artifact (dictionary) to a file."""
    if model_artifact:
        joblib.dump(model_artifact, 'pltv_model.pkl')
        print("Model artifact saved successfully.")

def retrain_and_save_model():
    """Loads data, trains the model, and saves the resulting artifact."""
    print("Loading raw event data...")
    raw_events = load_data()
    
    if not raw_events:
        return "No raw events found to train the model."
        
    print("Preprocessing data and calculating features for training...")
    features_df = preprocess_data_for_training(raw_events)
    
    if not features_df.empty and 'pltv' in features_df.columns and not features_df['pltv'].isnull().all():
        print("Training model with hyperparameter tuning...")
        model_artifact = train_model(features_df)
        
        if model_artifact:
            print("Saving model artifact...")
            save_model(model_artifact)
            return "Model training, validation, and saving process completed successfully."
        else:
            return "Model training failed."
    else:
        return "No data available to train the model (features DataFrame is empty or pLTV column is missing/empty)."

if __name__ == '__main__':
    message = retrain_and_save_model()
    print(message)

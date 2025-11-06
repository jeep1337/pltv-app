
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from database import get_all_customer_events, clear_customers_table
from features import calculate_features

def load_data():
    """Loads raw event data from the database."""
    events = get_all_customer_events()
    print("Raw events list from database:")
    # print(events) # This can be very verbose
    return events

def preprocess_data_for_training(raw_events):
    """Preprocesses the raw event data for model training."""
    all_individual_events = []
    for customer_id, event_data, _ in raw_events:
        if isinstance(event_data, dict) and 'events' in event_data and isinstance(event_data['events'], list):
            for event in event_data['events']:
                event_with_id = event.copy()
                event_with_id['customer_id'] = customer_id
                all_individual_events.append(event_with_id)

    if not all_individual_events:
        return pd.DataFrame()

    events_df = pd.DataFrame(all_individual_events)
    
    # Use the centralized feature calculation function
    customer_features_df = calculate_features(events_df)
    
    print("Customer Features DataFrame for Training:")
    print(customer_features_df)

    return customer_features_df

def train_model(df):
    """Trains the RandomForestRegressor model."""
    feature_columns = [
        'total_purchase_value', 'number_of_purchases', 'number_of_page_views',
        'days_since_last_purchase', 'purchase_frequency', 'average_purchase_value',
        'total_items_purchased', 'distinct_products_purchased', 'distinct_brands_purchased',
        'distinct_products_viewed', 'distinct_brands_viewed'
    ]
    
    # Ensure all feature columns exist, fill with 0 if not
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_columns]
    y = df['pltv']

    n_samples = len(df)

    # If the dataset is too small, skip grid search and train a default model
    if n_samples < 5:
        print("Warning: Dataset is too small for GridSearchCV. Training a default RandomForestRegressor.")
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Adjust cv to be at most the number of samples if it's greater
        cv = min(len(X_train), 3)
        if cv < 2:
            cv = 2 # cv must be at least 2

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Create a DataFrame for feature importances
    importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Model Feature Importances:")
    print(importances)

    return model

def save_model(model):
    """Saves the trained model to a file."""
    joblib.dump(model, 'pltv_model.pkl')

if __name__ == '__main__':
    # Optional: Clear the database for a fresh start
    # clear_customers_table()
    
    print("Loading raw event data...")
    raw_events = load_data()
    
    print("Preprocessing data and calculating features for training...")
    features_df = preprocess_data_for_training(raw_events)
    
    if not features_df.empty:
        print("Training model...")
        model = train_model(features_df)
        
        print("Saving model...")
        save_model(model)
        
        print("Model training and saving process completed successfully.")
    else:
        print("No data available to train the model.")

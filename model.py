import pandas as pd
import numpy as np
from database import get_all_customer_events, clear_customers_table
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_data():
    """Loads event data from the database and returns a pandas DataFrame."""
    events = get_all_customer_events()
    print("Raw events list from database:")
    print(events)
    df = pd.DataFrame(events, columns=['customer_id', 'event_data', 'created_at'])
    print("Raw DataFrame from database:")
    print(df)
    return df

def preprocess_data(df):
    print("Executing preprocess_data with event_name fix - v2")
    """Preprocesses the data and creates features for the model."""
    all_individual_events = []

    for index, row in df.iterrows():
        customer_id = row['customer_id']
        event_payload = row['event_data']

        if isinstance(event_payload, dict) and 'events' in event_payload and isinstance(event_payload['events'], list):
            for event in event_payload['events']:
                event_with_customer_id = event.copy()
                event_with_customer_id['customer_id'] = customer_id
                all_individual_events.append(event_with_customer_id)

    if not all_individual_events:
        return pd.DataFrame(columns=[
            'customer_id', 'total_purchase_value', 'number_of_purchases',
            'average_purchase_value', 'last_purchase_date', 'first_purchase_date',
            'number_of_page_views', 'days_since_last_purchase',
            'first_event_date', 'time_since_first_event', 'purchase_frequency', 'pltv'
        ])

    events_df = pd.DataFrame(all_individual_events)

    if 'value' not in events_df.columns:
        events_df['value'] = 0.0
    else:
        events_df['value'] = pd.to_numeric(events_df['value'], errors='coerce').fillna(0.0)

    if 'timestamp_micros' in events_df.columns:
        events_df['event_timestamp'] = pd.to_datetime(events_df['timestamp_micros'], unit='us', errors='coerce')
    elif 'request_start_time_ms' in events_df.columns:
        events_df['event_timestamp'] = pd.to_datetime(events_df['request_start_time_ms'], unit='ms', errors='coerce')
    elif 'api_timestamp_micros' in events_df.columns:
        events_df['event_timestamp'] = pd.to_datetime(events_df['api_timestamp_micros'], unit='us', errors='coerce')
    else:
        # If no timestamp is found, we can't process, so return an empty frame or handle appropriately
        print("Warning: No timestamp field found in event data. Supported fields are 'timestamp_micros', 'request_start_time_ms', or 'api_timestamp_micros'.")
        # Create an empty event_timestamp column to avoid crashing later
        events_df['event_timestamp'] = pd.NaT

    events_df['event_timestamp'] = events_df['event_timestamp'].dt.tz_localize(None)
    events_df = events_df.dropna(subset=['event_timestamp'])

    if 'event_name' in events_df.columns:
        purchases = events_df[events_df['event_name'] == 'purchase']
        page_views = events_df[events_df['event_name'] == 'page_view']
        event_name_col = 'event_name'
    elif 'event_type' in events_df.columns:
        print("Warning: 'event_name' not found, falling back to 'event_type'.")
        purchases = events_df[events_df['event_type'] == 'purchase']
        page_views = events_df[events_df['event_type'] == 'page_view']
        event_name_col = 'event_type'
    else:
        print("Error: Neither 'event_name' nor 'event_type' found in event data.")
        # Return an empty dataframe with the expected columns to avoid crashing
        return pd.DataFrame(columns=[
            'customer_id', 'total_purchase_value', 'number_of_purchases',
            'average_purchase_value', 'number_of_page_views', 'days_since_last_purchase',
            'purchase_frequency', 'time_since_first_event', 'pltv'
        ])

    purchase_features = purchases.groupby('customer_id').agg(
        total_purchase_value=('value', 'sum'),
        number_of_purchases=(event_name_col, 'count'),
        last_purchase_date=('event_timestamp', 'max'),
        first_purchase_date=('event_timestamp', 'min'),
    ).reset_index()

    purchase_features['average_purchase_value'] = purchase_features.apply(
        lambda row: row['total_purchase_value'] / row['number_of_purchases'] if row['number_of_purchases'] > 0 else 0,
        axis=1
    )

    page_view_features = page_views.groupby('customer_id').agg(
        number_of_page_views=(event_name_col, 'count')
    ).reset_index()

    all_customers = pd.DataFrame(df['customer_id'].unique(), columns=['customer_id'])
    customer_features = pd.merge(all_customers, purchase_features, on='customer_id', how='left')
    customer_features = pd.merge(customer_features, page_view_features, on='customer_id', how='left')

    current_date = pd.to_datetime('now').tz_localize(None)
    customer_features['last_purchase_date'] = customer_features['last_purchase_date'].fillna(pd.to_datetime('1970-01-01').tz_localize(None))
    customer_features['days_since_last_purchase'] = (current_date - customer_features['last_purchase_date']).dt.days

    customer_features['first_event_date'] = events_df.groupby('customer_id')['event_timestamp'].min().reset_index(name='first_event_date')['first_event_date']
    customer_features['time_since_first_event'] = (current_date - customer_features['first_event_date']).dt.days.fillna(0)

    customer_features['purchase_frequency'] = customer_features['number_of_purchases'] / customer_features['time_since_first_event']
    customer_features['purchase_frequency'] = customer_features['purchase_frequency'].replace([np.inf, -np.inf], 0).fillna(0)

    customer_features = customer_features.drop(columns=['last_purchase_date', 'first_purchase_date', 'first_event_date'])
    customer_features = customer_features.fillna(0)
    customer_features['pltv'] = customer_features['total_purchase_value']

    print("Customer Features DataFrame:")
    print(customer_features)

    return customer_features

def train_model(df):
    """Trains a simple linear regression model."""
    X = df[['total_purchase_value', 'number_of_purchases', 'number_of_page_views', 'days_since_last_purchase', 'purchase_frequency', 'average_purchase_value']]
    y = df['pltv']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    model = grid_search.best_estimator_

    # Print the best hyperparameters and feature importances
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Model Feature Importances: {model.feature_importances_}")

    return model

def save_model(model):
    """Saves the trained model to a file."""
    joblib.dump(model, 'pltv_model.pkl')

if __name__ == '__main__':
    # clear_customers_table() # Temporarily commented out for data loading
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)
    save_model(model)
    print("Model trained and saved successfully.")
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
    """Preprocesses the data and creates features for the model."""
    # Extract features from the event_data JSON
    df['event_name'] = df['event_data'].apply(lambda x: x.get('event_name'))
    df['value'] = df['event_data'].apply(lambda x: x.get('value', 0))
    df['event_timestamp'] = df['event_data'].apply(lambda x: pd.to_datetime(x.get('timestamp')).tz_localize(None))

    # Calculate features for each customer
    purchases = df[df['event_name'] == 'purchase']
    page_views = df[df['event_name'] == 'page_view']

    purchase_features = purchases.groupby('customer_id').agg(
        total_purchase_value=('value', 'sum'),
        number_of_purchases=('event_name', 'count'),
        last_purchase_date=('event_timestamp', 'max'),
        first_purchase_date=('event_timestamp', 'min'), # Get first purchase date
        average_purchase_value=('value', 'mean') # Calculate average purchase value
    ).reset_index()

    page_view_features = page_views.groupby('customer_id').agg(
        number_of_page_views=('event_name', 'count')
    ).reset_index()

    # Merge the features into a single DataFrame
    all_customers = pd.DataFrame(df['customer_id'].unique(), columns=['customer_id'])
    customer_features = pd.merge(all_customers, purchase_features, on='customer_id', how='left')
    customer_features = pd.merge(customer_features, page_view_features, on='customer_id', how='left')

    # Calculate days_since_last_purchase
    current_date = pd.to_datetime('now').tz_localize(None)
    # For customers with no purchases, last_purchase_date will be NaT. Fill with a very old date.
    customer_features['last_purchase_date'] = customer_features['last_purchase_date'].fillna(pd.to_datetime('1970-01-01').tz_localize(None))
    customer_features['days_since_last_purchase'] = (current_date - customer_features['last_purchase_date']).dt.days

    # Calculate time_since_first_event (customer tenure)
    customer_features['first_event_date'] = df.groupby('customer_id')['event_timestamp'].min().reset_index(name='first_event_date')['first_event_date']
    customer_features['time_since_first_event'] = (current_date - customer_features['first_event_date']).dt.days.fillna(0)

    # Calculate purchase_frequency
    # Avoid division by zero for new customers or customers with no events
    customer_features['purchase_frequency'] = customer_features['number_of_purchases'] / customer_features['time_since_first_event']
    customer_features['purchase_frequency'] = customer_features['purchase_frequency'].replace([np.inf, -np.inf], 0).fillna(0)

    # Drop intermediate date columns
    customer_features = customer_features.drop(columns=['last_purchase_date', 'first_purchase_date'])

    # Fill any remaining NaN values (e.g., for customers with no purchases or page views)
    customer_features = customer_features.fillna(0)

    # For now, we'll also create a dummy 'pltv' column for demonstration purposes.
    # In a real-world scenario, you would calculate this based on historical data.
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
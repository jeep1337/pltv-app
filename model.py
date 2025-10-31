import pandas as pd
from database import get_all_customer_events
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def load_data():
    """Loads event data from the database and returns a pandas DataFrame."""
    events = get_all_customer_events()
    df = pd.DataFrame(events, columns=['customer_id', 'event_data', 'created_at'])
    return df

def preprocess_data(df):
    """Preprocesses the data and creates features for the model."""
    # For now, we'll just count the number of events for each customer
    # and use that as a feature.
    df['event_count'] = df.groupby('customer_id')['customer_id'].transform('count')

    # In a real-world scenario, you would do more sophisticated feature
    # engineering here, such as:
    # - Recency, Frequency, Monetary (RFM) analysis
    # - Time-based features (e.g., time since last event)
    # - Features based on event types (e.g., number of purchases)

    # For now, we'll also create a dummy 'pltv' column for demonstration purposes.
    # In a real-world scenario, you would calculate this based on historical data.
    df['pltv'] = df['event_count'] * 10 # Dummy pLTV calculation

    return df

def train_model(df):
    """Trains a simple linear regression model."""
    X = df[['event_count']]
    y = df['pltv']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def save_model(model):
    """Saves the trained model to a file."""
    joblib.dump(model, 'pltv_model.pkl')

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)
    save_model(model)
    print("Model trained and saved successfully.")
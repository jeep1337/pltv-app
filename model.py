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
    # Extract features from the event_data JSON
    df['event_name'] = df['event_data'].apply(lambda x: x.get('event_name'))
    df['value'] = df['event_data'].apply(lambda x: x.get('value', 0))

    # Calculate features for each customer
    customer_features = df.groupby('customer_id').agg(
        total_purchase_value=('value', lambda x: x[df['event_name'] == 'purchase'].sum()),
        number_of_purchases=('event_name', lambda x: (x == 'purchase').sum()),
        number_of_page_views=('event_name', lambda x: (x == 'page_view').sum())
    ).reset_index()

    # For now, we'll also create a dummy 'pltv' column for demonstration purposes.
    # In a real-world scenario, you would calculate this based on historical data.
    customer_features['pltv'] = customer_features['total_purchase_value'] * 2 # Dummy pLTV calculation

    return customer_features

def train_model(df):
    """Trains a simple linear regression model."""
    X = df[['total_purchase_value', 'number_of_purchases', 'number_of_page_views']]
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
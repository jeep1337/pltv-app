
import pandas as pd
import numpy as np

def calculate_features(events_df):
    """
    Calculates features from a DataFrame of events.

    Args:
        events_df (pd.DataFrame): A DataFrame containing all events for one or more customers.

    Returns:
        pd.DataFrame: A DataFrame with aggregated features per customer.
    """
    if events_df.empty:
        return pd.DataFrame()

    # Ensure event_timestamp is present and correctly formatted
    if 'timestamp_micros' in events_df.columns:
        events_df['event_timestamp'] = pd.to_datetime(events_df['timestamp_micros'], unit='us', errors='coerce')
    elif 'request_start_time_ms' in events_df.columns:
        events_df['event_timestamp'] = pd.to_datetime(events_df['request_start_time_ms'], unit='ms', errors='coerce')
    elif 'api_timestamp_micros' in events_df.columns:
        events_df['event_timestamp'] = pd.to_datetime(events_df['api_timestamp_micros'], unit='us', errors='coerce')
    else:
        events_df['event_timestamp'] = pd.NaT

    events_df['event_timestamp'] = events_df['event_timestamp'].dt.tz_localize(None)
    events_df = events_df.dropna(subset=['event_timestamp'])

    # Determine the event name column
    if 'event_name' in events_df.columns:
        event_name_col = 'event_name'
    elif 'event_type' in events_df.columns:
        event_name_col = 'event_type'
    else:
        return pd.DataFrame() # Not enough data to process

    # Filter events
    purchases = events_df[events_df[event_name_col] == 'purchase']
    page_views = events_df[events_df[event_name_col] == 'page_view']
    view_events = events_df[events_df[event_name_col].isin(['view_item', 'add_to_cart'])]

    # --- Feature Aggregation ---

    # Purchase Features
    if 'value' not in purchases.columns:
        purchases['value'] = 0
        
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

    # Product-level Purchase Features
    product_events = []
    for _, purchase in purchases.iterrows():
        if 'items' in purchase and isinstance(purchase['items'], list):
            for item in purchase['items']:
                product_events.append({
                    'customer_id': purchase['customer_id'],
                    'item_id': item.get('item_id'),
                    'item_brand': item.get('item_brand'),
                    'quantity': item.get('quantity', 1)
                })
    if product_events:
        product_df = pd.DataFrame(product_events)
        product_features = product_df.groupby('customer_id').agg(
            total_items_purchased=('quantity', 'sum'),
            distinct_products_purchased=('item_id', 'nunique'),
            distinct_brands_purchased=('item_brand', 'nunique')
        ).reset_index()
    else:
        product_features = pd.DataFrame(columns=['customer_id', 'total_items_purchased', 'distinct_products_purchased', 'distinct_brands_purchased'])


    # Product View Features
    viewed_items = []
    for _, view_event in view_events.iterrows():
        if 'items' in view_event and isinstance(view_event['items'], list):
            for item in view_event['items']:
                viewed_items.append({
                    'customer_id': view_event['customer_id'],
                    'item_id': item.get('item_id'),
                    'item_brand': item.get('item_brand')
                })
    if viewed_items:
        viewed_items_df = pd.DataFrame(viewed_items)
        view_features = viewed_items_df.groupby('customer_id').agg(
            distinct_products_viewed=('item_id', 'nunique'),
            distinct_brands_viewed=('item_brand', 'nunique')
        ).reset_index()
    else:
        view_features = pd.DataFrame(columns=['customer_id', 'distinct_products_viewed', 'distinct_brands_viewed'])

    # Page View Features
    page_view_features = page_views.groupby('customer_id').agg(
        number_of_page_views=(event_name_col, 'count')
    ).reset_index()

    # --- Merge Features ---
    all_customers = pd.DataFrame(events_df['customer_id'].unique(), columns=['customer_id'])
    customer_features = pd.merge(all_customers, purchase_features, on='customer_id', how='left')
    customer_features = pd.merge(customer_features, product_features, on='customer_id', how='left')
    customer_features = pd.merge(customer_features, view_features, on='customer_id', how='left')
    customer_features = pd.merge(customer_features, page_view_features, on='customer_id', how='left')


    # --- Post-processing ---
    current_date = pd.to_datetime('now').tz_localize(None)
    customer_features['last_purchase_date'] = customer_features['last_purchase_date'].fillna(pd.to_datetime('1970-01-01').tz_localize(None))
    customer_features['days_since_last_purchase'] = (current_date - customer_features['last_purchase_date']).dt.days

    first_event_dates = events_df.groupby('customer_id')['event_timestamp'].min().reset_index()
    first_event_dates = first_event_dates.rename(columns={'event_timestamp': 'first_event_date'})
    customer_features = pd.merge(customer_features, first_event_dates, on='customer_id', how='left')

    customer_features['time_since_first_event'] = (current_date - customer_features['first_event_date']).dt.days.fillna(0)
    customer_features['time_since_first_event'] = customer_features['time_since_first_event'].apply(lambda x: max(x, 1))

    customer_features['purchase_frequency'] = customer_features['number_of_purchases'] / customer_features['time_since_first_event']
    customer_features['purchase_frequency'] = customer_features['purchase_frequency'].replace([np.inf, -np.inf], 0).fillna(0)

    # Drop intermediate columns and fill NaNs
    customer_features = customer_features.drop(columns=['last_purchase_date', 'first_purchase_date', 'first_event_date'])
    customer_features = customer_features.fillna(0)

    # Define pLTV
    customer_features['pltv'] = customer_features['total_purchase_value']

    return customer_features

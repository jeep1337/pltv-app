
import pandas as pd
import numpy as np
from functools import reduce

def calculate_features(events_df):
    """
    Calculates features from a DataFrame of events in a more efficient and robust way.

    Args:
        events_df (pd.DataFrame): A DataFrame containing all events for one or more customers.

    Returns:
        pd.DataFrame: A DataFrame with aggregated features per customer.
    """
    if events_df.empty:
        return pd.DataFrame()

    # --- Timestamp and Event Name Normalization ---
    for col in ['timestamp_micros', 'api_timestamp_micros']:
        if col in events_df.columns:
            events_df['event_timestamp'] = pd.to_datetime(events_df[col], unit='us', errors='coerce')
            break
    else:
        if 'request_start_time_ms' in events_df.columns:
            events_df['event_timestamp'] = pd.to_datetime(events_df['request_start_time_ms'], unit='ms', errors='coerce')
        else:
            events_df['event_timestamp'] = pd.NaT

    events_df = events_df.dropna(subset=['event_timestamp'])
    events_df['event_timestamp'] = pd.to_datetime(events_df['event_timestamp']).dt.tz_localize('UTC')

    event_name_col = 'event_name' if 'event_name' in events_df.columns else 'event_type'
    if event_name_col not in events_df.columns:
        return pd.DataFrame()

    # --- Efficiently Process Nested Item Data ---
    def extract_items(df, event_name):
        filtered_df = df[df[event_name_col] == event_name].copy()
        if 'items' not in filtered_df.columns:
            return pd.DataFrame()
        
        filtered_df = filtered_df.dropna(subset=['items'])
        exploded = filtered_df.explode('items')
        items_df = pd.json_normalize(exploded['items']).add_prefix('item.')
        return pd.concat([exploded.reset_index(drop=True), items_df], axis=1)

    purchase_items_df = extract_items(events_df, 'purchase')
    view_items_df = extract_items(events_df, 'view_item')
    add_to_cart_items_df = extract_items(events_df, 'add_to_cart')

    # --- Feature Aggregation using GroupBy ---
    all_customers = pd.DataFrame({'customer_id': events_df['customer_id'].unique()})

    # Purchase-based features
    purchases = events_df[events_df[event_name_col] == 'purchase'].copy()
    purchases['value'] = pd.to_numeric(purchases['value'], errors='coerce').fillna(0)
    purchase_features = purchases.groupby('customer_id').agg(
        total_purchase_value=('value', 'sum'),
        number_of_purchases=(event_name_col, 'size'),
        last_purchase_date=('event_timestamp', 'max'),
        first_purchase_date=('event_timestamp', 'min')
    ).reset_index()
    purchase_features['average_purchase_value'] = (purchase_features['total_purchase_value'] / purchase_features['number_of_purchases']).fillna(0)

    # Product-level features from purchases
    if not purchase_items_df.empty:
        purchase_items_df['item.quantity'] = pd.to_numeric(purchase_items_df['item.quantity'], errors='coerce').fillna(1)
        product_purchase_features = purchase_items_df.groupby('customer_id').agg(
            total_items_purchased=('item.quantity', 'sum'),
            distinct_products_purchased=('item.item_id', 'nunique'),
            distinct_brands_purchased=('item.item_brand', 'nunique')
        ).reset_index()
    else:
        product_purchase_features = pd.DataFrame(columns=['customer_id', 'total_items_purchased', 'distinct_products_purchased', 'distinct_brands_purchased'])

    # View-based features
    if not view_items_df.empty:
        view_features = view_items_df.groupby('customer_id').agg(
            distinct_products_viewed=('item.item_id', 'nunique'),
            distinct_brands_viewed=('item.item_brand', 'nunique')
        ).reset_index()
    else:
        view_features = pd.DataFrame(columns=['customer_id', 'distinct_products_viewed', 'distinct_brands_viewed'])

    # Other event counts
    event_counts = events_df.groupby(['customer_id', event_name_col]).size().unstack(fill_value=0)
    event_counts.columns = [f"{col}_count" for col in event_counts.columns]
    event_counts.rename(columns={'page_view_count': 'number_of_page_views'}, inplace=True)
    
    # --- Consolidate All Features ---
    dfs_to_merge = [
        all_customers,
        purchase_features,
        product_purchase_features,
        view_features,
        event_counts.reset_index()
    ]
    
    customer_features = reduce(lambda left, right: pd.merge(left, right, on='customer_id', how='left'), dfs_to_merge)

    # --- Post-processing and Final Feature Calculation ---
    current_date = pd.to_datetime('now', utc=True)
    
    customer_features['days_since_last_purchase'] = (current_date - customer_features['last_purchase_date']).dt.days
    
    first_event_dates = events_df.groupby('customer_id')['event_timestamp'].min().reset_index(name='first_event_date')
    customer_features = pd.merge(customer_features, first_event_dates, on='customer_id', how='left')
    
    customer_features['time_since_first_event'] = (current_date - customer_features['first_event_date']).dt.days.fillna(0).apply(lambda x: max(x, 1))
    
    customer_features['purchase_frequency'] = (customer_features['number_of_purchases'] / customer_features['time_since_first_event']).replace([np.inf, -np.inf], 0)

    # Define pLTV and clean up
    customer_features['pltv'] = customer_features['total_purchase_value']
    
    final_cols = [
        'customer_id', 'total_purchase_value', 'number_of_purchases', 'average_purchase_value',
        'total_items_purchased', 'distinct_products_purchased', 'distinct_brands_purchased',
        'distinct_products_viewed', 'distinct_brands_viewed', 'number_of_page_views',
        'days_since_last_purchase', 'time_since_first_event', 'purchase_frequency', 'pltv'
    ]
    
    # Add new event counts if they exist
    if 'add_to_cart_count' in event_counts.columns:
        final_cols.append('add_to_cart_count')
    if 'begin_checkout_count' in event_counts.columns:
        final_cols.append('begin_checkout_count')

    # Ensure all expected columns exist, filling missing ones with 0
    for col in final_cols:
        if col not in customer_features.columns:
            customer_features[col] = 0
            
    customer_features = customer_features[final_cols].fillna(0)

    return customer_features

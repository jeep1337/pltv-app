import json
import pandas as pd
from database import db
from features import calculate_features


def backfill():
    """
    Rebuilds customer_features from customer_events_normalized.
    Safe to run multiple times; uses upsert on customer_id.
    """
    rows = db.get_all_customer_events()  # [(customer_id, event_data, created_at), ...]
    events_by_customer = {}

    for customer_id, raw_event, _ in rows:
        evt = raw_event
        if isinstance(evt, str):
            try:
                evt = json.loads(evt)
            except Exception:
                continue
        if not isinstance(evt, dict):
            continue
        evt = dict(evt)
        evt.setdefault("customer_id", customer_id)
        if "event_name" in evt and isinstance(evt["event_name"], str):
            evt["event_name"] = evt["event_name"].lower()
        events_by_customer.setdefault(customer_id, []).append(evt)

    for customer_id, events in events_by_customer.items():
        if not events:
            continue
        df = pd.DataFrame(events)
        features_df = calculate_features(df)
        if features_df.empty:
            continue
        features_dict = features_df.iloc[0].to_dict()
        try:
            db.upsert_customer_features(features_dict)
        except Exception as exc:
            print(f"Failed to upsert features for {customer_id}: {exc}")


if __name__ == "__main__":
    backfill()

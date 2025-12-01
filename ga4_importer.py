
import os
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest

# Set up Google Analytics Data API client
# Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# to the path of your service account key file.
client = BetaAnalyticsDataClient()

PROPERTY_ID = "378715605"  # Your GA4 Property ID

def run_sample_report():
    """Runs a sample report using the Google Analytics Data API."""
    request = RunReportRequest(
        property=f"properties/{PROPERTY_ID}",
        date_ranges=[DateRange(start_date="60daysAgo", end_date="today")],
        dimensions=[
            Dimension(name="eventName"),
            Dimension(name="userPseudoId"),
            Dimension(name="eventTimestamp"),
            Dimension(name="itemId"),
            Dimension(name="itemName"),
            Dimension(name="itemBrand"),
            Dimension(name="itemCategory"),
            Dimension(name="pageLocation"),
        ],
        metrics=[
            Metric(name="eventCount"),
            Metric(name="value"),
            Metric(name="price"),
            Metric(name="quantity"),
        ],
        dimension_filter={
            "or_group": {
                "expressions": [
                    {"filter": {"field_name": "eventName", "string_filter": {"match_type": 1, "value": "purchase"}}},
                    {"filter": {"field_name": "eventName", "string_filter": {"match_type": 1, "value": "add_to_cart"}}},
                    {"filter": {"field_name": "eventName", "string_filter": {"match_type": 1, "value": "view_item"}}},
                    {"filter": {"field_name": "eventName", "string_filter": {"match_type": 1, "value": "begin_checkout"}}},
                    {"filter": {"field_name": "eventName", "string_filter": {"match_type": 1, "value": "page_view"}}},
                    {"filter": {"field_name": "eventName", "string_filter": {"match_type": 1, "value": "first_visit"}}},
                ]
            }
        }
    )
    response = client.run_report(request)

    print("Report result:")
    for row in response.rows:
        print(row.dimension_values[0].value, row.metric_values[0].value)

if __name__ == "__main__":
    run_sample_report()

from flask import Flask, request, jsonify
from database import connect_db, create_customers_table, get_all_customer_events
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get customer data from the request
    customer_data = request.get_json()

    # TODO: Use the model to predict pLTV

    # Return the prediction
    return jsonify({'pltv': 123.45})

@app.route('/event', methods=['POST'])
def event():
    event_data = request.get_json()
    customer_id = event_data.get('customer_id')

    if not customer_id:
        return jsonify({'error': 'customer_id is required'}), 400

    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO customers (customer_id, event_data) VALUES (%s, %s)",
                (customer_id, json.dumps(event_data))
            )
            conn.commit()
            return jsonify({'message': 'Event data stored successfully'}), 200
        except Exception as e:
            print(f"Error storing event data: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            conn.close()
    return jsonify({'error': 'Database connection failed'}), 500

@app.route('/events_data', methods=['GET'])
def events_data():
    events = get_all_customer_events()
    # Convert events to a more readable format if necessary
    formatted_events = []
    for event in events:
        formatted_events.append({
            "customer_id": event[0],
            "event_data": event[1],
            "created_at": event[2].isoformat() # Convert datetime to string
        })
    return jsonify(formatted_events), 200

if __name__ == '__main__':
    create_customers_table()
    app.run(debug=True)
from flask import Flask, request, jsonify
from database import connect_db, create_customers_table
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

if __name__ == '__main__':
    create_customers_table()
    app.run(debug=True)
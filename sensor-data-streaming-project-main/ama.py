from flask import Flask, jsonify, Response, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from confluent_kafka import Consumer, Producer
from joblib import load
import json
import threading
import os
import random
import time
import psycopg2
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# PostgreSQL setup
conn = psycopg2.connect(
    host=os.environ['PG_HOST'],
    database=os.environ['PG_DATABASE'],
    user=os.environ['PG_USER'],
    password=os.environ['PG_PASSWORD']
)
cursor = conn.cursor()

# Ensure tables exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    sensor_type VARCHAR(50),
    value DOUBLE PRECISION,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    sensor_type VARCHAR(50),
    value DOUBLE PRECISION,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

# Kafka configuration
kafka_conf = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'sensor-group',
    'auto.offset.reset': 'earliest'
}
producer = Producer({'bootstrap.servers': 'kafka:9092'})
consumer = Consumer(kafka_conf)
consumer.subscribe(['sensor_temperature', 'sensor_humidity', 'sensor_pressure'])

# Sensor simulation configuration
SENSOR_TOPICS = {
    'temperature': 'sensor_temperature',
    'humidity': 'sensor_humidity',
    'pressure': 'sensor_pressure'
}
UPDATE_INTERVALS = {
    'temperature': 2,
    'humidity': 3,
    'pressure': 4
}
SPIKE_PROBABILITY = 0.1

def generate_reading(sensor_type):
    ranges = {'temperature': (-45, 45), 'humidity': (30, 90), 'pressure': (10, 50)}
    return round(random.uniform(*ranges[sensor_type]), 2)

def should_spike():
    return random.random() < SPIKE_PROBABILITY

def delivery_report(err, msg):
    if err is not None:
        print(Fore.RED + 'Message delivery failed: {}'.format(err))
    else:
        print(Fore.GREEN + 'Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

def publish_sensor_data():
    last_update_time = {sensor: 0 for sensor in SENSOR_TOPICS}
    
    while True:
        current_time = time.time()
        for sensor, topic in SENSOR_TOPICS.items():
            if current_time - last_update_time[sensor] >= UPDATE_INTERVALS[sensor] or should_spike():
                data_value = generate_reading(sensor)
                sensor_record = {"sensor_type": sensor, "value": data_value, "timestamp": time.time()}

                # Save raw sensor data to PostgreSQL
                cursor.execute("""
                INSERT INTO sensor_data (sensor_type, value) VALUES (%s, %s)
                """, (sensor, data_value))
                conn.commit()

                # Produce to Kafka topic
                producer.produce(topic, json.dumps(sensor_record), callback=delivery_report)
                last_update_time[sensor] = current_time
                socketio.emit('sensor_data', {'type': sensor, 'value': data_value})
        
        producer.poll(0)
        time.sleep(0.1)

def predict_anomaly(model, value):
    prediction = model.predict([[value]])
    return prediction[0] == -1

def consume_and_process():
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue

        sensor_value = json.loads(msg.value().decode('utf-8'))
        sensor_type = msg.topic().split('_')[1]
        value = sensor_value[sensor_type]
        model = model_temperature if sensor_type == 'temperature' else model_humidity if sensor_type == 'humidity' else model_pressure

        if predict_anomaly(model, value):
            print(Fore.YELLOW + f"Anomaly detected in {sensor_type}: {value}")
            socketio.emit('anomaly_data', {'type': sensor_type, 'value': sensor_value})

            # Save anomaly to PostgreSQL
            cursor.execute("""
            INSERT INTO anomalies (sensor_type, value) VALUES (%s, %s)
            """, (sensor_type, value))
            conn.commit()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/sensor_historical_data')
def historical_data():
    return render_template("historical_data.html")

@app.route('/about_project')
def about_project():
    return render_template("about_project.html")

@app.route('/api/sensor_data/<sensor_type>', methods=['GET'])
def get_sensor_data(sensor_type):
    cursor.execute("""
    SELECT value, timestamp FROM sensor_data WHERE sensor_type = %s ORDER BY timestamp DESC LIMIT 100
    """, (sensor_type,))
    data = cursor.fetchall()
    return jsonify([{'value': row[0], 'timestamp': row[1]} for row in data])

@app.route('/api/anomalies/<sensor_type>', methods=['GET'])
def get_anomalies(sensor_type):
    cursor.execute("""
    SELECT value, timestamp FROM anomalies WHERE sensor_type = %s ORDER BY timestamp DESC LIMIT 100
    """, (sensor_type,))
    anomalies = cursor.fetchall()
    return jsonify([{'value': row[0], 'timestamp': row[1]} for row in anomalies])

if __name__ == '__main__':
    threading.Thread(target=consume_and_process, daemon=True).start()
    threading.Thread(target=publish_sensor_data, daemon=True).start()
    print("Running Flask with SocketIO")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

from flask import Flask, request, jsonify, Response
import requests
import os
import time
import psutil 
import pandas as pd
from joblib import load
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import mlflow
from mlflow.models.signature import infer_signature

app = Flask(__name__)

processor = load("../preprocessing/preprocessor.joblib")  

# pre processing column names
column_list = pd.read_csv("../preprocessing/column_list_processed.csv").columns.tolist()

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # Total request yang diterima
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # Waktu respons API
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # Throughput
 
# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # Penggunaan CPU
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # Penggunaan RAM

def transformed_data(raw_data):
    input_dataframe = pd.DataFrame([raw_data])
    transformed_data = processor.transform(input_dataframe).tolist()

    input_data =  {
        'dataframe_split': {
            'columns': column_list,
            'data': transformed_data
        }
    }

    return input_data

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  
    RAM_USAGE.set(psutil.virtual_memory().percent) 
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
 
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    REQUEST_COUNT.inc()  
    THROUGHPUT.inc()  

    MODEL_API_URL = os.getenv('MODEL_API_URL')
 
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "input data must be valid"}), 400

        cleaned_data = transformed_data(input_data)
        res = requests.post(MODEL_API_URL, headers=headers, json=cleaned_data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)  
        
        return jsonify({
            'Prediction:': res.json(),
            'Response time': f'{duration:.4f} seconds'
        }), 200
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)    

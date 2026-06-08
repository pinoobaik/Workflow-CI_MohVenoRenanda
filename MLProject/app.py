from flask import Flask, request, jsonify, Response
import psutil

from prometheus_client import Gauge
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from inference import predict

import time

app = Flask(__name__)

REQUEST_COUNT = Counter(
    "request_count",
    "Total Request"
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total Prediction"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request Latency"
)

CPU_USAGE = Gauge("system_cpu_usage", "System CPU Usage")
RAM_USAGE = Gauge("system_ram_usage", "System RAM Usage")

@app.route("/predict", methods=["POST"])
def prediction():

    REQUEST_COUNT.inc()

    start_time = time.time()

    data = request.json

    result = predict(data)

    PREDICTION_COUNT.inc()

    REQUEST_LATENCY.observe(
        time.time() - start_time
    )

    return jsonify({
        "prediction": result
    })

@app.route("/metrics")
def metrics():

    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
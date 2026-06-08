from flask import Flask, Response
import psutil

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

app = Flask(__name__)

REQUEST_COUNT = Counter("request_count_total", "Total Request")
PREDICTION_COUNT = Counter("prediction_count_total", "Total Prediction")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request Latency")

CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage")
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage")


@app.route("/metrics")
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
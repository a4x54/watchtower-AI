import logging
import os
import time
from typing import Dict

import psutil
import requests

API_KEY = os.getenv("API_KEY", "dev-key")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/report")
REPORT_INTERVAL = float(os.getenv("REPORT_INTERVAL", "3"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
LOGGER = logging.getLogger("anomaly-agent")


def collect_metrics() -> Dict[str, float]:
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    connections = len(psutil.net_connections(kind="inet"))
    net_io = psutil.net_io_counters()
    metrics = {
        "cpu": cpu_percent,
        "memory": memory_percent,
        "connections": connections,
        "bytes_sent": net_io.bytes_sent,
        "bytes_recv": net_io.bytes_recv,
    }
    return metrics


def send_report(session: requests.Session, metrics: Dict[str, float]) -> Dict:
    headers = {"X-API-Key": API_KEY}
    response = session.post(BACKEND_URL, json=metrics, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


def main() -> None:
    session = requests.Session()
    LOGGER.info("Starting agent. Sending data to %s", BACKEND_URL)
    try:
        while True:
            metrics = collect_metrics()
            LOGGER.info("Collected metrics: %s", metrics)
            try:
                result = send_report(session, metrics)
                LOGGER.info(
                    "Backend response | anomaly=%s | score=%.4f | decision=%.4f | threshold=%.4f",
                    result.get("is_anomaly"),
                    result.get("anomaly_score"),
                    result.get("decision_value"),
                    result.get("threshold", 0.0),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to send report: %s", exc)
            time.sleep(REPORT_INTERVAL)
    except KeyboardInterrupt:
        LOGGER.info("Agent stopped")


if __name__ == "__main__":
    main()

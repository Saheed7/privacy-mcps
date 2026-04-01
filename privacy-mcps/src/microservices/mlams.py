"""
ML Application Microservice (MLaMS) — Layer 6.

gRPC-based inference server that serves predictions from the trained model.
Run: python -m src.microservices.mlams
"""

import os
import numpy as np
import tensorflow as tf
import logging
from concurrent import futures
import grpc
import json

logger = logging.getLogger(__name__)

# If gRPC proto compiled, import them; otherwise provide a REST fallback
try:
    from src.microservices.proto import inference_pb2, inference_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("gRPC proto not compiled. Using REST fallback. "
                   "Run: python -m grpc_tools.protoc -I src/microservices/proto "
                   "--python_out=src/microservices/proto --grpc_python_out=src/microservices/proto "
                   "src/microservices/proto/inference.proto")


class MLaMSServer:
    """ML Application Microservice inference server."""

    def __init__(self, model_path: str, task: str = "binary"):
        self.task = task
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load trained model from disk."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        logger.info(f"Model loaded: {self.model.count_params()} parameters")

    def predict(self, features: list) -> dict:
        """Run inference on a single sample."""
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        probs = self.model.predict(x, verbose=0)

        if self.task == "binary":
            prob = float(probs[0][0])
            pred_class = 1 if prob >= 0.5 else 0
            label = "Malicious" if pred_class == 1 else "Benign"
            confidence = prob if pred_class == 1 else 1 - prob
            return {
                "predicted_class": pred_class,
                "confidence": confidence,
                "label": label,
                "probabilities": [1 - prob, prob],
            }
        else:
            pred_class = int(np.argmax(probs[0]))
            confidence = float(probs[0][pred_class])
            return {
                "predicted_class": pred_class,
                "confidence": confidence,
                "label": str(pred_class),
                "probabilities": probs[0].tolist(),
            }


def serve_rest(server: MLaMSServer, port: int = 8080):
    """Simple REST server fallback using http.server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/predict":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                result = server.predict(body["features"])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy", "model": server.model_path}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            logger.info(format % args)

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"MLaMS REST server listening on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="results/models/distributed_cic-iomt_binary_M6.h5")
    parser.add_argument("--task", default="binary")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = MLaMSServer(args.model_path, args.task)
    serve_rest(server, args.port)

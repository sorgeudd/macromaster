"""Minimal Flask test server for macro and sound management"""
import logging
import socket
import traceback
from flask import Flask, render_template
from flask_sock import Sock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestServer')

# Initialize Flask app
app = Flask(__name__)
sock = Sock(app)

@app.route('/')
def index():
    """Test endpoint"""
    return render_template('index.html')

@sock.route('/ws')
def websocket(ws):
    """Test WebSocket endpoint"""
    try:
        ws.send('Connected to WebSocket')
        while True:
            message = ws.receive()
            ws.send(f'Echo: {message}')
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        port = 5000
        logger.info(f"Testing port {port} availability...")
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(('0.0.0.0', port))
        test_socket.close()
        logger.info(f"Port {port} is available")

        logger.info(f"Starting test server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
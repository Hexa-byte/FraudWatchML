from app import app  # noqa: F401
import logging

if __name__ == '__main__':
    logging.info("Starting FraudWatch server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

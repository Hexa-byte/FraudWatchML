import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Initialize database
db = SQLAlchemy(model_class=Base)

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models here to ensure they're registered with SQLAlchemy
    import models  # noqa: F401
    
    # Create all tables
    logger.info("Creating database tables...")
    db.create_all()
    logger.info("Database tables created successfully.")

# Import routes after db initialization to avoid circular imports
from api import api_bp
from monitoring import monitoring_bp
from security import security_bp, init_app as init_security
import routes

# Initialize security (Flask-Login)
init_security(app)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api/v1')
app.register_blueprint(monitoring_bp, url_prefix='/monitoring')
app.register_blueprint(security_bp, url_prefix='/security')

logger.info("FraudWatch application initialized successfully.")

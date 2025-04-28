"""
Financial transaction data generation for FraudWatch

This module provides sample transaction data for testing and demonstrating
the fraud detection system. The data includes a mix of legitimate and fraudulent
transactions with realistic patterns.
"""

import os
import json
import random
import uuid
import csv
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from models import Transaction, ModelVersion, User, db

# Constants for data generation
MERCHANTS = [
    {"id": "merch_12345", "name": "Amazon", "category": "ecommerce", "risk_score": 0.1},
    {"id": "merch_23456", "name": "Target", "category": "retail", "risk_score": 0.2},
    {"id": "merch_34567", "name": "Best Buy", "category": "electronics", "risk_score": 0.3},
    {"id": "merch_45678", "name": "Shell", "category": "gas_station", "risk_score": 0.4},
    {"id": "merch_56789", "name": "Netflix", "category": "subscription", "risk_score": 0.1},
    {"id": "merch_67890", "name": "Uber", "category": "transportation", "risk_score": 0.2},
    {"id": "merch_78901", "name": "Unknown Vendor", "category": "unknown", "risk_score": 0.8},
    {"id": "merch_89012", "name": "International Payments Inc", "category": "financial", "risk_score": 0.7},
    {"id": "merch_90123", "name": "Virtual Assets Exchange", "category": "crypto", "risk_score": 0.6},
    {"id": "merch_01234", "name": "Cloudy Services LLC", "category": "digital_goods", "risk_score": 0.5}
]

PAYMENT_METHODS = ["card", "bank_transfer", "wallet", "crypto"]

CUSTOMER_PROFILES = [
    # Low-risk customers
    {"id_prefix": "cust_1", "count": 50, "avg_tx_count": 15, "fraud_probability": 0.01, "avg_amount": 150, "std_amount": 100},
    {"id_prefix": "cust_2", "count": 30, "avg_tx_count": 8, "fraud_probability": 0.02, "avg_amount": 500, "std_amount": 300},
    
    # Medium-risk customers
    {"id_prefix": "cust_3", "count": 15, "avg_tx_count": 5, "fraud_probability": 0.05, "avg_amount": 1000, "std_amount": 800},
    
    # High-risk customers (including fraudsters)
    {"id_prefix": "cust_4", "count": 5, "avg_tx_count": 3, "fraud_probability": 0.2, "avg_amount": 2000, "std_amount": 1500},
    {"id_prefix": "fraud_", "count": 3, "avg_tx_count": 2, "fraud_probability": 0.9, "avg_amount": 3000, "std_amount": 2000}
]

LOCATIONS = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "long": -74.0060},
    {"city": "Los Angeles", "country": "USA", "lat": 34.0522, "long": -118.2437},
    {"city": "Chicago", "country": "USA", "lat": 41.8781, "long": -87.6298},
    {"city": "London", "country": "UK", "lat": 51.5074, "long": -0.1278},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "long": 139.6503},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "long": 151.2093},
    {"city": "Moscow", "country": "Russia", "lat": 55.7558, "long": 37.6173},
    {"city": "Lagos", "country": "Nigeria", "lat": 6.5244, "long": 3.3792}
]

DEVICE_IDS = [f"dev_{i}" for i in range(100, 200)]
IP_PREFIXES = ["192.168.1.", "10.0.0.", "172.16.0.", "8.8.8."]

def generate_transaction(customer_id, fraud_probability, avg_amount, std_amount, timestamp=None):
    """Generate a single transaction record"""
    
    # Determine if this transaction is fraudulent based on probability
    is_fraud = random.random() < fraud_probability
    
    # Assign a transaction ID
    transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
    
    # Set timestamp if not provided
    if timestamp is None:
        timestamp = datetime.utcnow() - timedelta(days=random.randint(0, 30))
    
    # Higher amounts for fraudulent transactions
    amount_multiplier = 1.5 if is_fraud else 1.0
    amount = max(1.0, random.normalvariate(avg_amount * amount_multiplier, std_amount))
    
    # Select merchant (fraudulent transactions more likely to use high-risk merchants)
    if is_fraud and random.random() < 0.7:
        # Select from higher risk merchants
        merchant = random.choice(MERCHANTS[6:])
    else:
        merchant = random.choice(MERCHANTS)
    
    # Payment method (fraudulent transactions less likely to use bank transfers)
    if is_fraud:
        payment_method = random.choice(["card", "wallet", "crypto"])
    else:
        payment_method = random.choice(PAYMENT_METHODS)
    
    # Card present flag (fraudulent transactions usually card-not-present)
    card_present = False if is_fraud else (random.random() < 0.3)
    
    # Location data (fraudulent transactions more likely to use unusual locations)
    if is_fraud and random.random() < 0.6:
        location = random.choice(LOCATIONS[3:])  # More exotic locations
    else:
        location = random.choice(LOCATIONS[:4])  # More common locations
    
    # Device and IP (fraudulent transactions might use different devices)
    if is_fraud:
        device_id = random.choice(DEVICE_IDS[80:])  # Less common devices
        ip_address = f"{random.choice(IP_PREFIXES)}{random.randint(200, 255)}"
    else:
        device_id = random.choice(DEVICE_IDS[:80])
        ip_address = f"{random.choice(IP_PREFIXES)}{random.randint(1, 200)}"
    
    # Create the transaction record
    transaction = {
        "id": transaction_id,
        "amount": round(amount, 2),
        "timestamp": timestamp.isoformat(),
        "customer_id": customer_id,
        "merchant_id": merchant["id"],
        "payment_method": payment_method,
        "card_present": card_present,
        "is_fraud": is_fraud,
        "additional_features": {
            "ip_address": ip_address,
            "device_id": device_id,
            "location": {
                "latitude": location["lat"] + random.uniform(-0.01, 0.01),
                "longitude": location["long"] + random.uniform(-0.01, 0.01)
            },
            "merchant_category": merchant["category"],
            "merchant_name": merchant["name"],
            "merchant_risk_score": merchant["risk_score"],
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday()
        }
    }
    
    return transaction

def generate_dataset(num_days=30, output_file="transaction_data.csv"):
    """Generate a complete dataset of transactions"""
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=num_days)
    transactions = []
    
    print(f"Generating dataset for {num_days} days...")
    
    # Generate customer transactions
    for profile in CUSTOMER_PROFILES:
        for i in range(profile["count"]):
            customer_id = f"{profile['id_prefix']}{i+1000}"
            num_transactions = max(1, int(random.normalvariate(profile["avg_tx_count"], profile["avg_tx_count"]/3)))
            
            # Generate transactions for this customer
            for _ in range(num_transactions):
                tx_timestamp = start_date + timedelta(seconds=random.randint(0, num_days * 86400))
                tx = generate_transaction(
                    customer_id, 
                    profile["fraud_probability"],
                    profile["avg_amount"],
                    profile["std_amount"],
                    timestamp=tx_timestamp
                )
                transactions.append(tx)
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x["timestamp"])
    
    # Save to CSV
    if output_file:
        df = pd.DataFrame(transactions)
        # Flatten the additional_features for CSV
        for tx in transactions:
            for key, value in tx["additional_features"].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        df.loc[df['id'] == tx['id'], f"{key}_{subkey}"] = subvalue
                else:
                    df.loc[df['id'] == tx['id'], key] = value
        
        # Drop the additional_features column
        df = df.drop(columns=["additional_features"])
        df.to_csv(output_file, index=False)
        print(f"Saved {len(transactions)} transactions to {output_file}")
    
    return transactions

def load_transactions_to_db(transactions=None, file_path=None):
    """Load transaction data into the database"""
    if transactions is None and file_path:
        # Load from file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            transactions = df.to_dict('records')
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                transactions = json.load(f)
    
    if not transactions:
        print("No transactions to load")
        return
    
    print(f"Loading {len(transactions)} transactions to database...")
    
    # Convert to database model objects
    db_transactions = []
    for tx in transactions:
        # Convert timestamp string to datetime if needed
        timestamp = tx["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Extract location data
        latitude = None
        longitude = None
        if "additional_features" in tx and "location" in tx["additional_features"]:
            latitude = tx["additional_features"]["location"].get("latitude")
            longitude = tx["additional_features"]["location"].get("longitude")
        elif "location_latitude" in tx:
            latitude = tx["location_latitude"]
            longitude = tx["location_longitude"]
        
        # Extract other additional features
        ip_address = None
        device_id = None
        if "additional_features" in tx:
            ip_address = tx["additional_features"].get("ip_address")
            device_id = tx["additional_features"].get("device_id")
        elif "ip_address" in tx:
            ip_address = tx["ip_address"]
            device_id = tx["device_id"]
        
        # Create Transaction object
        db_tx = Transaction(
            id=tx["id"],
            amount=tx["amount"],
            timestamp=timestamp,
            customer_id=tx["customer_id"],
            merchant_id=tx["merchant_id"],
            payment_method=tx["payment_method"],
            card_present=tx["card_present"],
            ip_address=ip_address,
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            is_fraud=tx.get("is_fraud", False),
            fraud_score=tx.get("fraud_score", None)
        )
        db_transactions.append(db_tx)
    
    # Add to database in chunks to avoid memory issues
    chunk_size = 100
    for i in range(0, len(db_transactions), chunk_size):
        chunk = db_transactions[i:i+chunk_size]
        db.session.add_all(chunk)
        db.session.commit()
        print(f"Added {len(chunk)} transactions (batch {i//chunk_size + 1})")
    
    print(f"Successfully loaded {len(db_transactions)} transactions to database")

def create_sample_model_version():
    """Create a sample model version in the database"""
    # Check if a model version already exists
    existing = ModelVersion.query.first()
    if existing:
        print(f"Model version already exists: {existing}")
        return existing
    
    model = ModelVersion(
        version="v1.0.0",
        created_at=datetime.utcnow(),
        active=True,
        autoencoder_path="models/autoencoder_v1.h5",
        gbm_path="models/gbm_v1",
        precision=0.92,
        recall=0.89,
        created_by=1  # Assuming user ID 1 exists
    )
    
    db.session.add(model)
    db.session.commit()
    print(f"Created model version: {model}")
    return model

def create_admin_user():
    """Create an admin user if none exists"""
    # Check if any user exists
    existing = User.query.first()
    if existing:
        print(f"User already exists: {existing}")
        return existing
    
    admin = User(
        username="admin",
        email="admin@fraudwatch.com",
        role="admin",
        is_active=True
    )
    admin.set_password("admin123")  # In production, use a secure password
    
    db.session.add(admin)
    db.session.commit()
    print(f"Created admin user: {admin}")
    return admin

def initialize_sample_data(app, num_transactions=100):
    """Initialize the database with sample data"""
    with app.app_context():
        # Create admin user
        create_admin_user()
        
        # Create model version
        create_sample_model_version()
        
        # Generate and load transactions
        transactions = generate_dataset(num_days=30, output_file=None)
        if num_transactions and num_transactions < len(transactions):
            transactions = transactions[:num_transactions]
        
        load_transactions_to_db(transactions)
        
        print("Sample data initialization complete")


if __name__ == "__main__":
    # When run directly, generate a dataset file
    generate_dataset(num_days=30, output_file="transaction_data.csv")
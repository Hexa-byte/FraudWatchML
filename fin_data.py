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
import math
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from models import Transaction, ModelVersion, User, db

# Constants for data generation
MERCHANTS = [
    # Low-risk merchants
    {"id": "merch_10001", "name": "Amazon", "category": "ecommerce", "risk_score": 0.05},
    {"id": "merch_10002", "name": "Target", "category": "retail", "risk_score": 0.07},
    {"id": "merch_10003", "name": "Walmart", "category": "retail", "risk_score": 0.06},
    {"id": "merch_10004", "name": "Apple", "category": "electronics", "risk_score": 0.04},
    {"id": "merch_10005", "name": "Best Buy", "category": "electronics", "risk_score": 0.09},
    {"id": "merch_10006", "name": "Netflix", "category": "subscription", "risk_score": 0.03},
    {"id": "merch_10007", "name": "Spotify", "category": "subscription", "risk_score": 0.03},
    {"id": "merch_10008", "name": "Disney+", "category": "subscription", "risk_score": 0.02},
    {"id": "merch_10009", "name": "Costco", "category": "retail", "risk_score": 0.05},
    {"id": "merch_10010", "name": "Home Depot", "category": "home_improvement", "risk_score": 0.08},
    
    # Medium-risk merchants
    {"id": "merch_20001", "name": "Uber", "category": "transportation", "risk_score": 0.15},
    {"id": "merch_20002", "name": "Lyft", "category": "transportation", "risk_score": 0.17},
    {"id": "merch_20003", "name": "DoorDash", "category": "food_delivery", "risk_score": 0.19},
    {"id": "merch_20004", "name": "Grubhub", "category": "food_delivery", "risk_score": 0.18},
    {"id": "merch_20005", "name": "Shell", "category": "gas_station", "risk_score": 0.22},
    {"id": "merch_20006", "name": "Exxon", "category": "gas_station", "risk_score": 0.21},
    {"id": "merch_20007", "name": "Chevron", "category": "gas_station", "risk_score": 0.23},
    {"id": "merch_20008", "name": "Hilton Hotels", "category": "hospitality", "risk_score": 0.25},
    {"id": "merch_20009", "name": "Marriott", "category": "hospitality", "risk_score": 0.24},
    {"id": "merch_20010", "name": "United Airlines", "category": "travel", "risk_score": 0.27},
    
    # High-risk merchants
    {"id": "merch_30001", "name": "Online Casino Palace", "category": "gambling", "risk_score": 0.55},
    {"id": "merch_30002", "name": "BetWin Sports", "category": "gambling", "risk_score": 0.58},
    {"id": "merch_30003", "name": "Lucky Slots", "category": "gambling", "risk_score": 0.56},
    {"id": "merch_30004", "name": "Crypto Exchange Global", "category": "crypto", "risk_score": 0.62},
    {"id": "merch_30005", "name": "Bitcoin Direct", "category": "crypto", "risk_score": 0.65},
    {"id": "merch_30006", "name": "Digital Asset Trading", "category": "crypto", "risk_score": 0.63},
    {"id": "merch_30007", "name": "International Money Transfer", "category": "financial", "risk_score": 0.59},
    {"id": "merch_30008", "name": "Global Remittance Services", "category": "financial", "risk_score": 0.57},
    {"id": "merch_30009", "name": "Luxury Goods Emporium", "category": "luxury", "risk_score": 0.52},
    {"id": "merch_30010", "name": "Premium Collectibles", "category": "luxury", "risk_score": 0.54},
    
    # Very high-risk/suspicious merchants
    {"id": "merch_40001", "name": "Unknown Vendor LLC", "category": "unknown", "risk_score": 0.78},
    {"id": "merch_40002", "name": "Global Services Co", "category": "unknown", "risk_score": 0.82},
    {"id": "merch_40003", "name": "Offshore Investments", "category": "financial", "risk_score": 0.85},
    {"id": "merch_40004", "name": "Digital Wallet Services", "category": "financial", "risk_score": 0.79},
    {"id": "merch_40005", "name": "Anonymous Marketplace", "category": "digital_goods", "risk_score": 0.90}
]

PAYMENT_METHODS = ["card", "bank_transfer", "wallet", "crypto", "ach", "wire", "mobile_payment"]

# Common card BINs (Bank Identification Numbers) - fictional for simulation
CARD_BINS = {
    "visa": ["400000", "414720", "422222", "438123"],
    "mastercard": ["510000", "530000", "558800", "545454"],
    "amex": ["340000", "370000"],
    "discover": ["601100", "644000"]
}

CUSTOMER_PROFILES = [
    # Very low-risk customers (many transactions, small amounts)
    {"id_prefix": "cust_1", "count": 500, "avg_tx_count": 20, "fraud_probability": 0.002, "avg_amount": 75, "std_amount": 50},
    {"id_prefix": "cust_2", "count": 400, "avg_tx_count": 15, "fraud_probability": 0.005, "avg_amount": 150, "std_amount": 75},
    
    # Low-risk customers (regular activity)
    {"id_prefix": "cust_3", "count": 300, "avg_tx_count": 10, "fraud_probability": 0.01, "avg_amount": 300, "std_amount": 200},
    {"id_prefix": "cust_4", "count": 200, "avg_tx_count": 8, "fraud_probability": 0.015, "avg_amount": 500, "std_amount": 300},
    
    # Medium-risk customers
    {"id_prefix": "cust_5", "count": 100, "avg_tx_count": 6, "fraud_probability": 0.03, "avg_amount": 1000, "std_amount": 800},
    {"id_prefix": "cust_6", "count": 50, "avg_tx_count": 5, "fraud_probability": 0.04, "avg_amount": 1500, "std_amount": 1000},
    
    # High-risk customers
    {"id_prefix": "cust_7", "count": 25, "avg_tx_count": 4, "fraud_probability": 0.1, "avg_amount": 2000, "std_amount": 1500},
    {"id_prefix": "cust_8", "count": 15, "avg_tx_count": 3, "fraud_probability": 0.15, "avg_amount": 3000, "std_amount": 2000},
    
    # Known fraudsters
    {"id_prefix": "fraud_1", "count": 8, "avg_tx_count": 3, "fraud_probability": 0.7, "avg_amount": 2500, "std_amount": 2000},
    {"id_prefix": "fraud_2", "count": 5, "avg_tx_count": 2, "fraud_probability": 0.9, "avg_amount": 5000, "std_amount": 3000}
]

# More global locations
LOCATIONS = [
    # North America
    {"city": "New York", "country": "USA", "lat": 40.7128, "long": -74.0060, "region": "na"},
    {"city": "Los Angeles", "country": "USA", "lat": 34.0522, "long": -118.2437, "region": "na"},
    {"city": "Chicago", "country": "USA", "lat": 41.8781, "long": -87.6298, "region": "na"},
    {"city": "Houston", "country": "USA", "lat": 29.7604, "long": -95.3698, "region": "na"},
    {"city": "Phoenix", "country": "USA", "lat": 33.4484, "long": -112.0740, "region": "na"},
    {"city": "Toronto", "country": "Canada", "lat": 43.6532, "long": -79.3832, "region": "na"},
    {"city": "Mexico City", "country": "Mexico", "lat": 19.4326, "long": -99.1332, "region": "na"},
    
    # Europe
    {"city": "London", "country": "UK", "lat": 51.5074, "long": -0.1278, "region": "eu"},
    {"city": "Paris", "country": "France", "lat": 48.8566, "long": 2.3522, "region": "eu"},
    {"city": "Berlin", "country": "Germany", "lat": 52.5200, "long": 13.4050, "region": "eu"},
    {"city": "Madrid", "country": "Spain", "lat": 40.4168, "long": -3.7038, "region": "eu"},
    {"city": "Rome", "country": "Italy", "lat": 41.9028, "long": 12.4964, "region": "eu"},
    {"city": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "long": 4.9041, "region": "eu"},
    {"city": "Zurich", "country": "Switzerland", "lat": 47.3769, "long": 8.5417, "region": "eu"},
    
    # Asia-Pacific
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "long": 139.6503, "region": "ap"},
    {"city": "Shanghai", "country": "China", "lat": 31.2304, "long": 121.4737, "region": "ap"},
    {"city": "Beijing", "country": "China", "lat": 39.9042, "long": 116.4074, "region": "ap"},
    {"city": "Hong Kong", "country": "China", "lat": 22.3193, "long": 114.1694, "region": "ap"},
    {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "long": 103.8198, "region": "ap"},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "long": 151.2093, "region": "ap"},
    {"city": "Melbourne", "country": "Australia", "lat": -37.8136, "long": 144.9631, "region": "ap"},
    
    # Rest of World
    {"city": "Moscow", "country": "Russia", "lat": 55.7558, "long": 37.6173, "region": "row"},
    {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "long": -46.6333, "region": "row"},
    {"city": "Cairo", "country": "Egypt", "lat": 30.0444, "long": 31.2357, "region": "row"},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "long": 72.8777, "region": "row"},
    {"city": "Delhi", "country": "India", "lat": 28.7041, "long": 77.1025, "region": "row"},
    {"city": "Lagos", "country": "Nigeria", "lat": 6.5244, "long": 3.3792, "region": "row"},
    {"city": "Johannesburg", "country": "South Africa", "lat": -26.2041, "long": 28.0473, "region": "row"}
]

# Expanded Device IDs and browser/OS information
DEVICE_IDS = [f"dev_{i}" for i in range(100, 500)]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "Opera", "IE", "Samsung Browser", "UC Browser"]
OS_TYPES = ["Windows", "macOS", "iOS", "Android", "Linux", "Chrome OS"]
USER_AGENTS = [
    f"{browser}/{random.randint(40, 100)}.0.{random.randint(1000, 9999)}.{random.randint(10, 99)} ({os} {random.randint(8, 15)})"
    for browser in BROWSERS for os in OS_TYPES for _ in range(2)
]

# IP address ranges (fictional for simulation)
IP_PREFIXES = [
    "192.168.1.", "10.0.0.", "172.16.0.", "8.8.8.", "203.0.113.", "198.51.100.", 
    "192.0.2.", "100.64.0.", "169.254.0.", "240.0.0.", "64.233.160."
]

def generate_transaction(customer_id, fraud_probability, avg_amount, std_amount, timestamp=None):
    """Generate a single transaction record with realistic patterns"""
    
    # Determine if this transaction is fraudulent based on probability
    is_fraud = random.random() < fraud_probability
    
    # Assign a transaction ID with a format like tx_12a3b4c5d6e7
    transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
    
    # Set timestamp if not provided
    now = datetime.utcnow()
    if timestamp is None:
        # More recent transactions are more likely
        days_ago = int(random.betavariate(1, 3) * 30)  # Beta distribution favors recent days
        timestamp = now - timedelta(days=days_ago, 
                                   hours=random.randint(0, 23),
                                   minutes=random.randint(0, 59),
                                   seconds=random.randint(0, 59))
    
    # Time-of-day patterns (fraud more common late at night)
    hour_of_day = timestamp.hour
    is_business_hours = 8 <= hour_of_day <= 18
    is_late_night = hour_of_day >= 22 or hour_of_day <= 4
    
    # Amount generation - different patterns based on fraud and time
    if is_fraud:
        # Fraudulent transactions: either very small (testing) or larger than normal
        if random.random() < 0.2:  # 20% chance of "test" transaction
            amount = random.uniform(0.5, 10.0)  # Small "test" transaction
        else:
            # Larger amounts for fraudulent transactions
            amount_multiplier = random.uniform(1.2, 2.5)
            amount = max(1.0, random.normalvariate(avg_amount * amount_multiplier, std_amount * 1.2))
            
            # Round to common values for large amounts
            if amount > 1000:
                amount = round(amount / 100) * 100  # Round to nearest 100
    else:
        # Legitimate transaction
        # Business hours typically have larger transactions
        time_multiplier = 1.2 if is_business_hours else 0.9
        amount = max(1.0, random.normalvariate(avg_amount * time_multiplier, std_amount))
        
        # Round to realistic amounts (people tend to spend rounded amounts)
        if amount < 20:
            # Small amounts often end in .99 or .95
            amount = math.floor(amount) + random.choice([0, 0.49, 0.5, 0.95, 0.99])
        elif amount < 100:
            # Medium amounts often rounded to nearest 5 or 10
            amount = round(amount / 5) * 5
    
    # Select merchant based on fraud risk and other patterns
    if is_fraud:
        if random.random() < 0.7:
            # High risk merchants for fraudulent transactions
            merchant_indices = list(range(len(MERCHANTS)))
            # Weight toward riskier merchants
            weights = [m["risk_score"] for m in MERCHANTS]
            merchant_idx = random.choices(merchant_indices, weights=weights, k=1)[0]
            merchant = MERCHANTS[merchant_idx]
        else:
            # Sometimes fraudsters use normal merchants to avoid detection
            merchant = random.choice(MERCHANTS[:20])  # Low to medium risk merchants
    else:
        # Legitimate transactions mostly use reputable merchants
        if random.random() < 0.85:
            # Most legitimate transactions use low-medium risk merchants
            merchant = random.choice(MERCHANTS[:20])
        else:
            # Occasionally legitimate users shop at higher risk merchants
            merchant = random.choice(MERCHANTS)
    
    # Payment method selection
    if is_fraud:
        # Fraudulent transactions avoid bank transfers (higher security)
        if random.random() < 0.8:
            payment_method = random.choice(["card", "wallet", "crypto"])
        else:
            # Sometimes fraudsters try other methods
            payment_method = random.choice(PAYMENT_METHODS)
            
        # Card-present flag (fraudulent transactions usually card-not-present)
        card_present = random.random() < 0.1  # Only 10% are card present
    else:
        # Legitimate payment methods follow different patterns based on amount
        if amount < 50:
            # Small amounts often card or mobile
            payment_method = random.choice(["card", "mobile_payment", "wallet"])
        elif amount < 500:
            # Medium amounts use various methods
            payment_method = random.choice(PAYMENT_METHODS)
        else:
            # Large amounts more likely to use bank methods
            payment_method = random.choice(["bank_transfer", "wire", "card", "ach"])
            
        # Card-present more common for legitimate transactions
        card_present = random.random() < 0.4  # 40% are card present
    
    # Generate card data if payment method is card
    card_data = None
    if payment_method == "card":
        card_type = random.choice(list(CARD_BINS.keys()))
        bin_number = random.choice(CARD_BINS[card_type])
        # Generate rest of card number (don't store full numbers in real systems!)
        masked_number = f"{bin_number}xxxxxx{random.randint(1000, 9999)}"
        card_data = {
            "type": card_type,
            "bin": bin_number,
            "masked_number": masked_number,
            "exp_year": now.year + random.randint(0, 5),
            "exp_month": random.randint(1, 12)
        }
    
    # Location data with realistic patterns
    if is_fraud:
        if random.random() < 0.6:
            # Fraudulent transactions often from suspicious locations
            location = random.choice(LOCATIONS[14:])  # More exotic locations
            
            # Sometimes location jumps (impossible travel)
            if "last_location" in globals() and random.random() < 0.5:
                # Calculate distance to simulate impossible travel (different continent)
                regions = [loc["region"] for loc in LOCATIONS if loc["region"] != last_location["region"]]
                if regions:
                    target_region = random.choice(regions)
                    possible_locations = [loc for loc in LOCATIONS if loc["region"] == target_region]
                    if possible_locations:
                        location = random.choice(possible_locations)
        else:
            # Sometimes fraudsters use common locations to blend in
            location = random.choice(LOCATIONS[:7])  # Common locations
    else:
        # Legitimate transactions often from common locations
        if random.random() < 0.85:
            # Most legitimate transactions from common locations
            location = random.choice(LOCATIONS[:14])  # More common locations
        else:
            # Sometimes legitimate users travel
            location = random.choice(LOCATIONS)
    
    # Store last location for impossible travel detection
    globals()["last_location"] = location
    
    # Add some random noise to coordinates for realism
    lat_noise = random.uniform(-0.01, 0.01)
    long_noise = random.uniform(-0.01, 0.01)
    
    # Device and IP address selection
    user_agent = random.choice(USER_AGENTS)
    
    if is_fraud:
        # Fraudulent transactions might use suspicious devices or IPs
        device_id = random.choice(DEVICE_IDS[300:])  # Less common devices
        ip_address = f"{random.choice(IP_PREFIXES[5:])}{random.randint(200, 255)}"
    else:
        # Legitimate users tend to use common devices
        device_id = random.choice(DEVICE_IDS[:300])
        ip_address = f"{random.choice(IP_PREFIXES[:5])}{random.randint(1, 200)}"
    
    # Transaction velocity features
    # These would be calculated in real-time in a production system
    # Here we generate them synthetically
    tx_velocity_24h = random.randint(1, 20)
    tx_velocity_1h = random.randint(0, 5)
    
    if is_fraud:
        # Fraudsters often have unusual velocity patterns
        if random.random() < 0.6:
            tx_velocity_1h = random.randint(3, 15)  # High 1-hour velocity
    
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
            "user_agent": user_agent,
            "location": {
                "latitude": location["lat"] + lat_noise,
                "longitude": location["long"] + long_noise,
                "city": location["city"],
                "country": location["country"],
                "region": location["region"]
            },
            "merchant_category": merchant["category"],
            "merchant_name": merchant["name"],
            "merchant_risk_score": merchant["risk_score"],
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_weekend": timestamp.weekday() >= 5,
            "is_business_hours": is_business_hours,
            "is_late_night": is_late_night,
            "tx_velocity_24h": tx_velocity_24h,
            "tx_velocity_1h": tx_velocity_1h
        }
    }
    
    # Add card data if available
    if card_data:
        transaction["additional_features"]["card_data"] = card_data
    
    return transaction

def generate_dataset(num_days=30, output_file=None):
    """Generate a complete dataset of transactions
    
    Args:
        num_days (int): Number of days to generate data for
        output_file (str, optional): File path to save CSV data, or None to skip saving
        
    Returns:
        list: Generated transaction data
    """
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=num_days)
    transactions = []
    
    # Initialize last_location for impossible travel detection
    if "last_location" not in globals():
        globals()["last_location"] = LOCATIONS[0]
    
    print(f"Generating dataset for {num_days} days...")
    
    # Monitor fraud ratio to ensure realistic dataset (target ~1-2% fraud)
    target_fraud_ratio = 0.015  # 1.5% fraud rate
    
    # Generate customer transactions
    for profile in CUSTOMER_PROFILES:
        for i in range(profile["count"]):
            customer_id = f"{profile['id_prefix']}{i+1000}"
            
            # Scale up transaction count a bit for more data
            # Low-risk customers have more transactions
            tx_count_multiplier = 1.0
            if "cust_1" in profile["id_prefix"] or "cust_2" in profile["id_prefix"]:
                tx_count_multiplier = 2.0  # More transactions for very low-risk customers
            elif "fraud" in profile["id_prefix"]:
                tx_count_multiplier = 0.8  # Fewer transactions for fraudsters
                
            # Use normal distribution with slight skew for transaction count
            base_count = max(1, profile["avg_tx_count"] * tx_count_multiplier)
            num_transactions = max(1, int(random.normalvariate(base_count, base_count/4)))
            
            # Add transaction bursts for some customers
            if random.random() < 0.1:  # 10% of customers have burst periods
                num_transactions += random.randint(3, 10)
            
            # Generate transactions for this customer
            for _ in range(num_transactions):
                # Create time distribution weighted toward more recent days (customers more active recently)
                days_ago = int(random.betavariate(1, 2) * num_days)
                tx_timestamp = end_date - timedelta(days=days_ago, 
                                                  hours=random.randint(0, 23),
                                                  minutes=random.randint(0, 59),
                                                  seconds=random.randint(0, 59))
                
                # Generate the transaction with appropriate risk profile
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
    
    # Calculate and report fraud ratio
    fraud_count = sum(1 for tx in transactions if tx["is_fraud"])
    fraud_ratio = fraud_count / len(transactions)
    print(f"Generated {len(transactions)} transactions with {fraud_count} fraudulent ({fraud_ratio:.2%})")
    
    # Save to CSV if output file specified
    if output_file:
        try:
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
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
    
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

def initialize_sample_data(app, num_transactions=10000):
    """Initialize the database with sample data
    
    Args:
        app: Flask application instance
        num_transactions (int): Target number of transactions to generate (default: 10000)
    """
    with app.app_context():
        print(f"Initializing database with target {num_transactions} transactions...")
        
        # Create admin user
        create_admin_user()
        
        # Create model version
        create_sample_model_version()
        
        # Generate and load transactions
        # Use 90 days to get more data for realistic patterns
        transactions = generate_dataset(num_days=90, output_file=None)
        
        # Check if we have enough transactions
        if len(transactions) < num_transactions:
            print(f"Only generated {len(transactions)} transactions, which is less than requested {num_transactions}")
            print("Using all available transactions")
        elif len(transactions) > num_transactions:
            print(f"Generated {len(transactions)} transactions, trimming to requested {num_transactions}")
            # Prioritize keeping more recent transactions and fraud cases
            # First, separate fraud and non-fraud
            fraud_txs = [tx for tx in transactions if tx["is_fraud"]]
            normal_txs = [tx for tx in transactions if not tx["is_fraud"]]
            
            # Keep all fraud transactions if possible (they're more interesting)
            if len(fraud_txs) <= num_transactions:
                # We can keep all fraud and some non-fraud
                num_normal_to_keep = num_transactions - len(fraud_txs)
                # Sort normal transactions by recency (newest first)
                normal_txs.sort(key=lambda x: x["timestamp"], reverse=True)
                # Keep the most recent ones
                normal_txs = normal_txs[:num_normal_to_keep]
                # Combine fraud and selected normal transactions
                transactions = fraud_txs + normal_txs
            else:
                # Too many fraud transactions, need to sample
                # Keep all normal transactions and sample from fraud
                if len(normal_txs) < num_transactions:
                    num_fraud_to_keep = num_transactions - len(normal_txs)
                    # Randomly sample from fraud transactions
                    random.shuffle(fraud_txs)
                    fraud_txs = fraud_txs[:num_fraud_to_keep]
                    transactions = normal_txs + fraud_txs
                else:
                    # Too many transactions overall, just take the most recent ones
                    transactions.sort(key=lambda x: x["timestamp"], reverse=True)
                    transactions = transactions[:num_transactions]
        
        # Sort by timestamp before loading
        transactions.sort(key=lambda x: x["timestamp"])
        
        # Load to database in chunks
        load_transactions_to_db(transactions)
        
        print(f"Sample data initialization complete with {len(transactions)} transactions")


if __name__ == "__main__":
    # When run directly, generate a dataset file
    generate_dataset(num_days=30, output_file="transaction_data.csv")
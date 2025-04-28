"""
Initialize FraudWatch database with sample data

This script populates the database with:
1. Admin user
2. Sample model version
3. Transaction data (mix of legitimate and fraudulent)

Usage:
    python initialize_data.py [num_transactions]
    
    num_transactions: Number of transactions to generate (default: 100)
"""

import sys
import os
from app import app
from fin_data import initialize_sample_data

if __name__ == "__main__":
    # Get number of transactions from command line argument, default to 100
    num_transactions = 100
    if len(sys.argv) > 1:
        try:
            num_transactions = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of transactions: {sys.argv[1]}")
            print("Using default value: 100")
    
    print(f"Initializing FraudWatch database with {num_transactions} sample transactions...")
    initialize_sample_data(app, num_transactions)
    print("Initialization complete. You can now login with:")
    print("  Username: admin")
    print("  Password: admin123")
#!/bin/bash

# Setup script for FraudWatch ML project
echo "Setting up FraudWatch ML environment..."

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create the models directory if it doesn't exist
echo "Creating necessary directories..."
mkdir -p models

# Setup database
echo "Setting up database..."
if [ -z "$DATABASE_URL" ]; then
  echo "Warning: DATABASE_URL environment variable not set."
  echo "Please set your PostgreSQL database connection string as DATABASE_URL."
  echo "Example: export DATABASE_URL=postgresql://username:password@localhost:5432/fraudwatch"
else
  echo "DATABASE_URL environment variable is set."
fi

# Initialize the database with sample data
echo "Would you like to initialize the database with sample data? (y/n)"
read -r answer
if [ "$answer" = "y" ]; then
  echo "Initializing database with sample data..."
  python initialize_data.py
else
  echo "Skipping database initialization."
fi

echo "Setup complete! You can now run the application with:"
echo "python main.py"
echo "or"
echo "gunicorn --bind 0.0.0.0:5000 main:app"
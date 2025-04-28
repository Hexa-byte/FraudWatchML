#!/bin/bash

# This script will help you upload the FraudWatch code to your GitHub repository
# Make sure you have Git installed and you're logged in to GitHub

echo "Starting upload process for FraudWatch project to GitHub..."

# Clone the repository
echo "Cloning your repository..."
git clone https://github.com/Hexa-byte/FraudWatchML.git
cd FraudWatchML

# Copy all the project files
echo "Copying project files..."
cp -r ../api.py ../app.py ../config.py ../data_pipeline.py ../fin_data.py \
      ../initialize_data.py ../main.py ../ml_pipeline.py ../model_architecture.py \
      ../models.py ../monitoring.py ../routes.py ../security.py ../utils.py \
      ../README.md ./

# Create necessary directories if they don't exist
mkdir -p static templates tests

# Copy additional files if they exist
if [ -d "../static" ]; then
  cp -r ../static/* ./static/
fi

if [ -d "../templates" ]; then
  cp -r ../templates/* ./templates/
fi

if [ -d "../tests" ]; then
  cp -r ../tests/* ./tests/
fi

# Create requirements.txt file
echo "Creating requirements.txt..."
cat > requirements.txt << EOF
# Python dependencies for FraudWatch
Flask==2.3.3
Flask-Login==0.6.2
Flask-SQLAlchemy==3.1.1
SQLAlchemy==2.0.23
Werkzeug==2.3.7
pandas==2.1.1
numpy==1.25.2
scipy==1.11.3
matplotlib==3.8.0
tensorflow==2.14.0
tensorflow-decision-forests==1.5.0
gunicorn==21.2.0
email-validator==2.0.0
psycopg2-binary==2.9.9
EOF

# Add all files to git
echo "Adding files to Git..."
git add .

# Commit the changes
echo "Committing changes..."
git commit -m "Initial commit of FraudWatch ML project"

# Push to GitHub
echo "Pushing to GitHub repository..."
git push

echo "Upload complete! Check your repository at https://github.com/Hexa-byte/FraudWatchML"
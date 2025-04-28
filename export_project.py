#!/usr/bin/env python3
"""
Export the FraudWatch ML project as a ZIP file
This script creates a ZIP archive of the current project, excluding unnecessary files.
"""

import os
import sys
import zipfile
from datetime import datetime

def should_include(file_path):
    """Check if a file should be included in the export"""
    # Exclude these directories
    excluded_dirs = [
        '__pycache__', 
        'venv', 
        'env', 
        '.git', 
        '.ipynb_checkpoints',
        '.vscode',
        '.idea'
    ]
    
    # Exclude these file patterns
    excluded_patterns = [
        '.pyc', 
        '.pyo', 
        '.pyd', 
        '.db', 
        '.sqlite', 
        '.sqlite3',
        '.env',
        '.DS_Store',
        '.coverage',
        'transaction_data.csv'
    ]
    
    # Check excluded directories
    parts = file_path.split(os.sep)
    for excluded_dir in excluded_dirs:
        if excluded_dir in parts:
            return False
    
    # Check excluded patterns
    for pattern in excluded_patterns:
        if file_path.endswith(pattern):
            return False
    
    return True

def create_zip(output_path):
    """Create a ZIP file with the project contents"""
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to always include
    essential_files = [
        'api.py',
        'app.py',
        'config.py',
        'data_pipeline.py',
        'fin_data.py',
        'initialize_data.py',
        'main.py',
        'ml_pipeline.py',
        'model_architecture.py',
        'models.py',
        'monitoring.py',
        'routes.py',
        'security.py',
        'utils.py',
        'README.md',
        'requirements.txt',
        '.gitignore',
        'setup.sh'
    ]
    
    # Create ZIP file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add essential files first
        for file_name in essential_files:
            file_path = os.path.join(root_dir, file_name)
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
                print(f"Added: {file_name}")
        
        # Walk through directories and add other files
        for root, dirs, files in os.walk(root_dir):
            # Process only specific directories
            for dir_name in ['templates', 'static', 'tests']:
                dir_path = os.path.join(root_dir, dir_name)
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    for subroot, subdirs, subfiles in os.walk(dir_path):
                        for file in subfiles:
                            file_path = os.path.join(subroot, file)
                            if should_include(file_path):
                                rel_path = os.path.relpath(file_path, root_dir)
                                zipf.write(file_path, rel_path)
                                print(f"Added: {rel_path}")

if __name__ == "__main__":
    # Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fraudwatch_export_{timestamp}.zip"
    
    create_zip(output_file)
    print(f"\nExport complete! File created: {output_file}")
    print(f"Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
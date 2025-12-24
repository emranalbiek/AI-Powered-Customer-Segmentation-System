#!/bin/bash

echo "=========================================="
echo "Customers Segmentation - Setup"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv customers-segmentation

# Activate virtual environment
echo "Activating virtual environment..."
source customers-segmentation/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p artifacts

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
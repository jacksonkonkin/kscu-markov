#!/bin/bash

# Simple launch script for KSCU Wallet-Share Predictor
echo "🏦 Launching KSCU Wallet-Share Predictor..."

# Activate virtual environment
source venv/bin/activate

# Launch Streamlit app
streamlit run prototype/app.py --server.port=8501
#!/usr/bin/env python3
"""
Launch script for the KSCU Wallet-Share Markov Challenge prototype.
This script ensures all dependencies are met and launches the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # Package name vs import name
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data():
    """Check if data files exist."""
    data_file = Path("data/raw/KSCU_wallet_share_train.xls")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the dataset is in the correct location")
        return False
    
    print(f"✅ Data file found: {data_file}")
    return True

def launch_streamlit():
    """Launch the Streamlit application."""
    try:
        print("🚀 Launching KSCU Wallet-Share Predictor...")
        print("📊 The prototype will open in your web browser")
        print("🔄 Press Ctrl+C to stop the application")
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "prototype/app.py",
            "--server.port=8502",
            "--server.address=localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main function to run all checks and launch the app."""
    print("🏦 KSCU Wallet-Share Markov Challenge - Prototype Launcher")
    print("=" * 60)
    
    # Check working directory
    if not Path("prototype/app.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Run checks
    if not check_requirements():
        sys.exit(1)
    
    if not check_data():
        sys.exit(1)
    
    # Launch application
    launch_streamlit()

if __name__ == "__main__":
    main()
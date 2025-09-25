"""Configuration and constants for KSCU Markov model."""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "KSCU_wallet_share_train.xls"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# State definitions
STATES = ['STAY', 'SPLIT', 'LEAVE']

# State assignment thresholds
STATE_THRESHOLDS = {
    'LEAVE': (0.0, 0.2),
    'SPLIT': (0.2, 0.8),
    'STAY': (0.8, 1.0)
}

# Feature groups
DEMOGRAPHIC_FEATURES = ['age', 'tenure_years']

FINANCIAL_FEATURES = ['avg_balance', 'product_count', 'has_mortgage', 
                      'card_spend_monthly']

ENGAGEMENT_FEATURES = ['digital_engagement', 'branch_visits_last_q', 
                       'salary_deposit_flag']

RISK_FEATURES = ['complaints_12m', 'fee_events_12m', 'rate_sensitivity']

MARKETING_FEATURES = ['promo_exposure', 'nps_bucket']

# All features for modeling
MODEL_FEATURES = (DEMOGRAPHIC_FEATURES + FINANCIAL_FEATURES + 
                 ENGAGEMENT_FEATURES + RISK_FEATURES + MARKETING_FEATURES)

# Target variables
TARGET_STATE = 'next_state'
TARGET_WALLET_SHARE = 'wallet_share_next'

# Model hyperparameters
MARKOV_PARAMS = {
    'smoothing_alpha': 0.01,  # Laplace smoothing parameter
    'min_samples_leaf': 50,   # Minimum samples for state assignment
}

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'log_loss']
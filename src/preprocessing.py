"""Data preprocessing and feature engineering for KSCU Markov model."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE, SPLITS_DIR,
        STATE_THRESHOLDS, MODEL_FEATURES, 
        TARGET_STATE, TARGET_WALLET_SHARE,
        RANDOM_STATE, TEST_SIZE, VAL_SIZE
    )
except ImportError:
    from config import (
        RAW_DATA_FILE, PROCESSED_DATA_FILE, SPLITS_DIR,
        STATE_THRESHOLDS, MODEL_FEATURES, 
        TARGET_STATE, TARGET_WALLET_SHARE,
        RANDOM_STATE, TEST_SIZE, VAL_SIZE
    )


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw dataset from file."""
    if file_path is None:
        file_path = RAW_DATA_FILE
    
    # The file is actually CSV despite .xls extension
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def assign_state(wallet_share: float) -> str:
    """Assign state based on wallet share value."""
    if wallet_share < STATE_THRESHOLDS['LEAVE'][1]:
        return 'LEAVE'
    elif wallet_share < STATE_THRESHOLDS['SPLIT'][1]:
        return 'SPLIT'
    else:
        return 'STAY'


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for modeling."""
    df = df.copy()
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 30, 40, 50, 60, 100],
                             labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    
    # Tenure groups
    df['tenure_group'] = pd.cut(df['tenure_years'], 
                                bins=[0, 2, 5, 10, 100],
                                labels=['0-2y', '2-5y', '5-10y', '10+y'])
    
    # Balance categories
    df['balance_category'] = pd.qcut(df['avg_balance'], 
                                     q=4, 
                                     labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Product engagement score (combined metric)
    df['product_engagement'] = (
        df['product_count'] * 0.4 +
        df['digital_engagement'] * 0.3 +
        (100 - df['branch_visits_last_q'] * 10) * 0.3  # Favor digital over branch
    )
    
    # Risk score (higher = more risky)
    df['risk_score'] = (
        df['complaints_12m'] * 0.3 +
        df['fee_events_12m'] * 0.2 +
        df['rate_sensitivity'] * 0.5
    )
    
    # Wallet share change
    df['wallet_share_delta'] = df['wallet_share_next'] - df['wallet_share']
    
    # State change indicator
    df['state_changed'] = (df['state'] != df['next_state']).astype(int)
    
    # Life stage (simplified)
    df['life_stage'] = df.apply(lambda x: 
        'Young' if x['age'] < 30 else
        'Family' if x['age'] < 50 and x['has_mortgage'] else
        'Mature' if x['age'] < 65 else
        'Senior', axis=1)
    
    # Customer value segment
    df['value_segment'] = df.apply(lambda x:
        'High' if x['avg_balance'] > 50000 and x['product_count'] >= 3 else
        'Medium' if x['avg_balance'] > 20000 or x['product_count'] >= 2 else
        'Low', axis=1)
    
    # Digital adoption level
    df['digital_adoption'] = pd.cut(df['digital_engagement'],
                                    bins=[0, 20, 50, 80, 100],
                                    labels=['None', 'Low', 'Medium', 'High'])
    
    # Quarter as numeric (for trend analysis)
    quarter_map = {'t0': 0, 't1': 1, 't2': 2, 't3': 3}
    df['quarter_numeric'] = df['quarter'].map(quarter_map)
    
    return df


def prepare_transition_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Markov transition analysis."""
    df = df.copy()
    
    # Sort by customer and quarter
    df = df.sort_values(['customer_id', 'quarter_numeric'])
    
    # For each customer, identify transitions
    transitions = []
    
    for customer_id in df['customer_id'].unique():
        customer_data = df[df['customer_id'] == customer_id].sort_values('quarter_numeric')
        
        for i in range(len(customer_data) - 1):
            current_row = customer_data.iloc[i]
            next_row = customer_data.iloc[i + 1]
            
            # Verify sequential quarters
            if next_row['quarter_numeric'] == current_row['quarter_numeric'] + 1:
                transition = {
                    'customer_id': customer_id,
                    'from_quarter': current_row['quarter'],
                    'to_quarter': next_row['quarter'],
                    'from_state': current_row['state'],
                    'to_state': next_row['state'],
                    'from_wallet_share': current_row['wallet_share'],
                    'to_wallet_share': next_row['wallet_share'],
                    **{f: current_row[f] for f in MODEL_FEATURES if f in current_row}
                }
                transitions.append(transition)
    
    transition_df = pd.DataFrame(transitions)
    print(f"Created {len(transition_df):,} transition records")
    return transition_df


def split_data(df: pd.DataFrame, 
               stratify_col: str = 'state') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df[stratify_col] if stratify_col else None
    )
    
    # Second split: train vs val
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE,
        stratify=train_val[stratify_col] if stratify_col else None
    )
    
    print(f"Data split - Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def scale_features(train: pd.DataFrame, 
                  val: pd.DataFrame, 
                  test: pd.DataFrame,
                  feature_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    
    # Fit on train, transform all
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    
    return train_scaled, val_scaled, test_scaled


def preprocess_pipeline(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complete preprocessing pipeline."""
    
    # Load data
    df = load_raw_data(file_path)
    
    # Create features
    df = create_features(df)
    
    # Split data
    train, val, test = split_data(df)
    
    # Save processed splits
    train.to_csv(SPLITS_DIR / 'train.csv', index=False)
    val.to_csv(SPLITS_DIR / 'val.csv', index=False)
    test.to_csv(SPLITS_DIR / 'test.csv', index=False)
    
    print(f"\nSaved processed splits to {SPLITS_DIR}")
    print(f"Features created: {len(df.columns) - 20} new features")
    
    return train, val, test


if __name__ == "__main__":
    # Run preprocessing
    train, val, test = preprocess_pipeline()
    
    # Display summary
    print("\n" + "="*50)
    print("Preprocessing Complete!")
    print("="*50)
    print(f"\nTrain shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    
    print("\nState distributions:")
    print("Train:", train['state'].value_counts(normalize=True).round(3).to_dict())
    print("Val:  ", val['state'].value_counts(normalize=True).round(3).to_dict())
    print("Test: ", test['state'].value_counts(normalize=True).round(3).to_dict())
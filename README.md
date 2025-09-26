# KSCU Wallet-Share Markov Challenge

Competition entry for the Kootenay Savings Credit Union (KSCU) Wallet-Share Markov Challenge.

**Deadline: September 29, 2025 (11:59 PM PT)**

## Project Structure

```
kscu/
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Processed data
│   └── splits/             # Train/val/test splits
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory data analysis
├── src/
│   ├── config.py          # Configuration
│   ├── preprocessing.py   # Data preprocessing
│   └── markov_model.py    # Markov chain model
├── requirements.txt       # Dependencies
└── CLAUDE.md             # Detailed project guide
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Explore the Data

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/01_eda.ipynb
```

### 3. Run Preprocessing

```bash
# Process the data and create train/val/test splits
python -m src.preprocessing
```

### 4. Train Markov Model

```bash
# Train the base Markov chain model
python -m src.markov_model
```

## Dataset Overview

- **20,000 records** (5,000 customers × 4 quarters)
- **20 features** including demographics, financial metrics, and engagement scores
- **3 states**: STAY (82.6%), SPLIT (10.8%), LEAVE (6.6%)
- **Target**: Predict state transitions and wallet share

## Key Features

| Feature | Description |
|---------|------------|
| customer_id | Unique identifier |
| age | Customer age (18-84) |
| tenure_years | Years as KSCU member |
| product_count | Number of products held |
| avg_balance | Average account balance |
| digital_engagement | Digital channel usage (0-100) |
| wallet_share | Current wallet share (0-1) |
| state | Current state (STAY/SPLIT/LEAVE) |
| next_state | Next quarter state |
| wallet_share_next | Next quarter wallet share |

## State Definitions

- **STAY**: wallet_share ≥ 0.80 (high loyalty)
- **SPLIT**: 0.20 ≤ wallet_share < 0.80 (mixed banking)
- **LEAVE**: wallet_share < 0.20 (at risk/churned)

## Key Insights from EDA

1. **Strong retention**: 96% of STAY customers remain in STAY state
2. **Risk indicators**: Lower digital engagement, fewer products, more complaints
3. **Positive factors**: Product count, digital engagement, and balance strongly correlate with wallet share

## Next Steps

1. Data exploration and preprocessing
2. Basic Markov model implementation
3. Feature engineering and model refinement
4. Build interactive prototype (Streamlit)
5. Generate business insights and report

## Competition Scoring

- **60%** - Predictive Quality (LogLoss, wallet-share error)
- **25%** - Business Value & Rigor (insights, fairness)
- **15%** - Application & Delivery (prototype, presentation)

## Contact

- Competition: jamie.gummo@kscu.com
- Documentation: See CLAUDE.md for detailed instructions
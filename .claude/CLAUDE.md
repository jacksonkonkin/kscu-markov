# CLAUDE.md - KSCU Wallet-Share Markov Challenge

## Project Overview

This project is a competition entry for the Kootenay Savings Credit Union (KSCU) Wallet-Share Markov Challenge. The goal is to build an AI-powered Markov chain solution that predicts member behavior and wallet share in retail banking.

**Competition Deadline: September 25, 2025 (11:59 PM PT)**

## Project Context

### Business Problem
KSCU needs to understand and predict member behavior across three states:
- **Stay**: Member maintains all banking with KSCU (wallet share ≈ 1.0)
- **Split**: Member uses KSCU and other institutions (0 < wallet share < 1.0)  
- **Leave**: Member closes KSCU accounts (wallet share ≈ 0)

### Competition Requirements
1. **Prediction Model**: Estimate transition probabilities between states
2. **Forecasting**: Predict expected wallet share (0-1 scale)
3. **Hypothesis Testing**: Identify key drivers of member behavior
4. **Prototype**: Interactive AI agent for scenario testing

### Scoring Weights
- Predictive Quality: 60% (LogLoss, wallet-share error, calibration)
- Business Value & Rigor: 25% (insights, fairness, stability)
- Application & Delivery: 15% (prototype usability, presentation clarity)

## Project Structure

```
kscu-markov-challenge/
├── data/
│   ├── raw/                 # Original dataset (arrives Sept 18)
│   ├── processed/           # Cleaned and feature-engineered data
│   └── splits/             # Train/validation/test splits
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_features.ipynb  # Feature engineering
│   ├── 03_markov.ipynb    # Markov model development
│   └── 04_evaluation.ipynb # Model evaluation
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration and constants
│   ├── preprocessing.py   # Data cleaning and preparation
│   ├── features.py        # Feature engineering
│   ├── markov_model.py    # Core Markov chain implementation
│   ├── evaluation.py      # Metrics and validation
│   └── utils.py          # Helper functions
├── prototype/
│   ├── app.py            # Streamlit/Gradio application
│   ├── components.py     # UI components
│   └── scenarios.py      # Scenario testing logic
├── tests/
│   ├── test_markov.py
│   └── test_features.py
├── reports/
│   ├── technical_report.tex  # LaTeX source
│   ├── executive_summary.md
│   └── figures/
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Technical Stack

### Core Dependencies
```python
# Data processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=1.7.0  # For feature importance

# Markov chains
hmmlearn>=0.3.0  # Hidden Markov Models
pymc>=5.0.0  # Probabilistic programming

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Prototype
streamlit>=1.25.0
plotly>=5.14.0

# Development
jupyter>=1.0.0
pytest>=7.3.0
black>=23.0.0
```

## Data Schema

### Input Features (Expected in dataset)
- `member_id`: Unique identifier
- `quarter`: Time period (1-6)
- `age`: Member age
- `tenure`: Months as KSCU member
- `balance_checking`: Checking account balance
- `balance_savings`: Savings account balance
- `has_mortgage`: Binary flag
- `mortgage_balance`: Mortgage amount (if applicable)
- `digital_engagement`: Engagement score (0-100)
- `num_products`: Count of KSCU products
- `wallet_share`: Current wallet share (0-1)

### Derived Features to Create
- `wallet_share_delta`: Quarter-over-quarter change
- `engagement_trend`: Rolling average of digital engagement
- `product_adoption_rate`: New products per quarter
- `balance_concentration`: Ratio of largest to total balance
- `life_stage`: Derived from age and products

## Key Algorithms and Methods

### 1. State Assignment Logic
```python
def assign_state(wallet_share):
    if wallet_share >= 0.8:
        return 'Stay'
    elif wallet_share > 0.2:
        return 'Split'
    else:
        return 'Leave'
```

### 2. Transition Matrix Calculation
- Count transitions between quarters
- Normalize by row sums
- Apply Laplace smoothing for zero counts

### 3. Feature-Dependent Transitions
- Use logistic regression for transition probabilities
- Incorporate member features as covariates
- Time-varying transition matrices

### 4. Wallet Share Forecasting
- Regression model within each state
- Weighted by transition probabilities
- Monte Carlo simulation for confidence intervals

## Development Workflow

### Phase 1: Data Preparation (Sept 18-19)
1. Load and validate dataset
2. Perform EDA and create visualizations
3. Engineer features
4. Create train/validation/test splits (60/20/20)

### Phase 2: Model Development (Sept 19-21)
1. Build baseline Markov model
2. Implement feature-dependent transitions
3. Develop wallet share forecasting
4. Run hypothesis tests on key drivers

### Phase 3: Prototype Creation (Sept 21-23)
1. Design UI/UX for scenario testing
2. Implement interactive visualizations
3. Add intervention simulation capabilities
4. Test with example scenarios

### Phase 4: Documentation (Sept 23-25)
1. Write technical report (≤6 pages)
2. Create executive summary (≤2 pages)
3. Document code and ensure reproducibility
4. Final testing and submission preparation

## Code Standards

### Style Guidelines
- Use Black for Python formatting
- Type hints for all functions
- Docstrings following NumPy style
- Maximum line length: 88 characters

### Testing Requirements
- Unit tests for all core functions
- Integration tests for end-to-end pipeline
- Validation on holdout data
- Cross-validation for stability

### Reproducibility
- Set random seeds explicitly
- Log all preprocessing steps
- Version control data transformations
- Document all assumptions

## Key Tasks for Claude Code

### High Priority Tasks
1. **EDA Analysis**: "Analyze the dataset and create comprehensive visualizations showing member behavior patterns, state transitions, and wallet share distributions across quarters"

2. **Markov Model Implementation**: "Implement a Markov chain model with feature-dependent transition probabilities using the member characteristics as covariates"

3. **Prototype Development**: "Create a Streamlit application that allows users to input member characteristics and see predicted state transitions and wallet share forecasts"

### Common Commands

```bash
# Data exploration
python -m src.preprocessing --explore data/raw/dataset.csv

# Train model
python -m src.markov_model --train --config config.yaml

# Run evaluation
python -m src.evaluation --model models/best_model.pkl

# Launch prototype
streamlit run prototype/app.py

# Run tests
pytest tests/ -v

# Generate report figures
python -m src.visualization --output reports/figures/
```

## Scenario Testing Examples

### Business Scenarios to Implement
1. **Digital Adoption Campaign**: Increase digital_engagement by 20 points
2. **Mortgage Promotion**: Offer mortgages to high-balance members
3. **Youth Banking**: Target members under 30 with product bundles
4. **Retention Program**: Identify at-risk members (Split→Leave probability > 0.3)

### Intervention Impact Metrics
- Change in average wallet share
- Reduction in attrition rate
- Expected revenue impact
- Cost-benefit analysis

## Performance Optimization

### Model Performance Targets
- LogLoss < 0.5 for state prediction
- Wallet share MAE < 0.15
- Calibration error < 0.1
- F1-score > 0.7 for Leave prediction

### Computational Efficiency
- Vectorize transition calculations
- Use sparse matrices where applicable
- Implement caching for repeated computations
- Parallel processing for simulations

## Risk Considerations

### Model Risks
- Overfitting to 6 quarters of data
- Temporal drift in member behavior
- Class imbalance (few Leave cases)
- Feature leakage from future information

### Business Risks
- Fairness across demographic groups
- Interpretability for non-technical users
- Actionability of recommendations
- Regulatory compliance (fair lending)

## Submission Checklist

- [ ] Model generates transition probabilities
- [ ] Wallet share forecasts for all members
- [ ] Technical report (≤6 pages, PDF)
- [ ] Executive summary (≤2 pages, PDF)
- [ ] Reproducible code with requirements.txt
- [ ] Prototype application (working locally)
- [ ] README with setup instructions
- [ ] All code runs offline
- [ ] Random seeds set for reproducibility
- [ ] No external API dependencies

## Contact and Resources

**Competition Contact**: Jamie Gummo (jamie.gummo@kscu.com)

**Key Dates**:
- Dataset Release: September 18, 2025
- Submission Deadline: September 25, 2025, 11:59 PM PT
- Winner Announcement: September 29, 2025

## Notes for Claude Code

When working on this project:
1. Always validate data transformations with small samples first
2. Create visualizations to verify model behavior
3. Test edge cases (new members, extreme wallet shares)
4. Focus on business interpretability over complex models
5. Ensure all code is deterministic and reproducible
6. Comment complex logic thoroughly
7. Prioritize the 60% scoring weight on predictive quality
8. Remember this is for a co-op position - show learning ability


## Success Criteria

The winning solution will:
- Accurately predict member state transitions
- Provide actionable business insights
- Include a user-friendly prototype
- Demonstrate robust methodology
- Show consideration for fairness and stability
- Be fully reproducible offline
- Clearly communicate technical and business value
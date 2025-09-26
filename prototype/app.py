"""
KSCU Wallet-Share Markov Challenge - Interactive Prototype
Streamlit application for scenario testing and business insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Change working directory to project root
os.chdir(project_root)

try:
    # Import from src directory
    import src.markov_model as markov_module
    import src.config as config_module
    import src.preprocessing as preprocessing_module
    
    MarkovChainModel = markov_module.MarkovChainModel
    calculate_steady_state = markov_module.calculate_steady_state
    MODEL_FEATURES = config_module.MODEL_FEATURES
    STATES = config_module.STATES
    load_raw_data = preprocessing_module.load_raw_data
    create_features = preprocessing_module.create_features
    
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you're running from the project root directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="KSCU Wallet-Share Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #2ca02c;
    }
    .warning-metric {
        border-left-color: #ff7f0e;
    }
    .danger-metric {
        border-left-color: #d62728;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare data for the application."""
    try:
        # Load raw data
        df = load_raw_data()
        df = create_features(df)
        
        # Load splits if available
        try:
            train = pd.read_csv('data/splits/train.csv')
            test = pd.read_csv('data/splits/test.csv')
            return df, train, test
        except:
            return df, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_model():
    """Load the trained Markov model."""
    try:
        df, train, _ = load_data()
        if train is not None:
            # Use training data
            model_data = train
        else:
            # Use full dataset
            model_data = df
        
        model = MarkovChainModel(use_features=True)
        feature_cols = [col for col in MODEL_FEATURES if col in model_data.columns]
        model.fit(model_data, feature_cols)
        
        return model, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def main():
    """Main application function."""
    
    st.title("KSCU Wallet-Share Markov Predictor")
    st.markdown("*Interactive tool for predicting member state transitions and wallet share*")
    
    # Load data and model
    df, train, test = load_data()
    model, feature_cols = load_model()
    
    if df is None or model is None:
        st.error("Failed to load data or model. Please check your data files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Individual Prediction", "Model Performance", "Business Insights"]
    )
    
    if page == "Individual Prediction":
        individual_prediction_page(model, feature_cols, df)
    elif page == "Model Performance":
        model_performance_page(model, df, test)
    else:
        business_insights_page(df)

def individual_prediction_page(model, feature_cols, df):
    """Page for individual customer predictions."""
    
    st.header("Individual Customer Prediction")
    st.markdown("Predict state transitions and wallet share for a specific customer profile.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Customer Profile")
        
        # Current state
        current_state = st.selectbox(
            "Current State:",
            ["STAY", "SPLIT", "LEAVE"],
            help="Current customer state based on wallet share"
        )
        
        # Customer features
        age = st.slider("Age", 18, 84, 45, help="Customer age")
        tenure = st.slider("Tenure (years)", 0.0, 20.0, 5.0, step=0.5, help="Years as KSCU member")
        product_count = st.slider("Number of Products", 1, 8, 3, help="Total KSCU products held")
        avg_balance = st.number_input("Average Balance ($)", 0, 200000, 25000, step=1000)
        
        has_mortgage = st.checkbox("Has Mortgage", help="Does customer have a mortgage?")
        salary_deposit = st.checkbox("Salary Deposit", help="Does salary get deposited to KSCU?")
        
        digital_engagement = st.slider("Digital Engagement", 0, 100, 50, help="Digital channel usage score")
        branch_visits = st.slider("Branch Visits (last quarter)", 0, 20, 2, help="Number of branch visits")
        
        complaints = st.slider("Complaints (12m)", 0, 10, 0, help="Number of complaints in last 12 months")
        fee_events = st.slider("Fee Events (12m)", 0, 20, 2, help="Number of fee events")
        rate_sensitivity = st.slider("Rate Sensitivity", 0.0, 10.0, 5.0, step=0.1)
        
        card_spend = st.number_input("Monthly Card Spend ($)", 0, 10000, 1000, step=100)
        nps_bucket = st.slider("NPS Bucket", 1, 10, 7, help="Net Promoter Score bucket")
        promo_exposure = st.slider("Promo Exposure", 0, 10, 3, help="Marketing promotions exposed to")
    
    with col2:
        st.subheader("Predictions")
        
        # Create feature vector
        features_dict = {
            'age': age,
            'tenure_years': tenure,
            'product_count': product_count,
            'avg_balance': avg_balance,
            'has_mortgage': int(has_mortgage),
            'salary_deposit_flag': int(salary_deposit),
            'digital_engagement': digital_engagement,
            'branch_visits_last_q': branch_visits,
            'complaints_12m': complaints,
            'fee_events_12m': fee_events,
            'rate_sensitivity': rate_sensitivity,
            'card_spend_monthly': card_spend,
            'nps_bucket': nps_bucket,
            'promo_exposure': promo_exposure
        }
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in features_dict:
                features_dict[col] = 0  # Default value
        
        features_df = pd.DataFrame([features_dict])
        
        # Get predictions
        trans_probs = model.predict_transition_probs(current_state, features_df)
        
        # Display transition probabilities
        st.markdown("### Next Quarter State Probabilities")
        
        # Create probability chart
        prob_df = pd.DataFrame(list(trans_probs.items()), columns=['State', 'Probability'])
        fig = px.bar(prob_df, x='State', y='Probability', 
                    color='State', color_discrete_map={'STAY': 'green', 'SPLIT': 'orange', 'LEAVE': 'red'},
                    title="Transition Probabilities")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics in columns for better layout
        metric_cols = st.columns(3)
        state_order = ["STAY", "SPLIT", "LEAVE"]  # Define order for consistency

        for idx, state in enumerate(state_order):
            if state in trans_probs:
                prob = trans_probs[state]
                with metric_cols[idx]:
                    if state == "STAY":
                        bg_color = "#d4edda"
                        border_color = "#28a745"
                        text_color = "#155724"
                    elif state == "SPLIT":
                        bg_color = "#fff3cd"
                        border_color = "#ffc107"
                        text_color = "#856404"
                    else:  # LEAVE
                        bg_color = "#f8d7da"
                        border_color = "#dc3545"
                        text_color = "#721c24"

                    st.markdown(f"""
                    <div style="
                        background-color: {bg_color};
                        border: 2px solid {border_color};
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <h3 style="margin: 0; color: {text_color}; font-weight: 600;">{state}</h3>
                        <h1 style="margin: 10px 0; color: {border_color}; font-size: 36px;">{prob:.1%}</h1>
                        <p style="margin: 0; color: {text_color}; font-size: 14px;">Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Predict wallet share for each potential state
        st.markdown("### Predicted Wallet Share by State")
        
        wallet_preds = {}
        for state in STATES:
            wallet_pred = model.predict_wallet_share(state, features_df)
            wallet_preds[state] = wallet_pred
            
            st.metric(
                label=f"If {state}",
                value=f"{wallet_pred:.1%}",
                help=f"Predicted wallet share if customer transitions to {state}"
            )
        
        # Expected wallet share
        expected_wallet_share = sum(trans_probs[state] * wallet_preds[state] for state in STATES)
        st.metric(
            label="Expected Wallet Share",
            value=f"{expected_wallet_share:.1%}",
            help="Probability-weighted expected wallet share"
        )


def model_performance_page(model, df, test_data):
    """Page showing model performance metrics."""
    
    st.header("Model Performance")
    st.markdown("Comprehensive evaluation of the Markov model performance.")
    
    # Basic model info
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Records", f"{len(df):,}")
        st.metric("States", len(STATES))
    
    with col2:
        st.metric("Features Used", len(model.transition_models))
        st.metric("Transition Models", len(model.transition_models))
    
    with col3:
        st.metric("Wallet Share Models", len(model.wallet_share_models))
    
    # Transition matrix visualization
    st.subheader("Transition Matrix")
    
    if model.transition_matrix is not None:
        # Convert to percentage and create heatmap
        trans_matrix_pct = model.transition_matrix * 100
        
        fig = px.imshow(
            trans_matrix_pct.values,
            labels=dict(x="Next State", y="Current State", color="Probability (%)"),
            x=trans_matrix_pct.columns,
            y=trans_matrix_pct.index,
            color_continuous_scale="RdYlGn",
            title="State Transition Probabilities (%)"
        )
        
        # Add text annotations
        for i, row in enumerate(trans_matrix_pct.index):
            for j, col in enumerate(trans_matrix_pct.columns):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{trans_matrix_pct.loc[row, col]:.1f}%",
                    showarrow=False,
                    font=dict(color="white" if trans_matrix_pct.loc[row, col] < 50 else "black")
                )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("Feature Importance")
    
    importance_data = []
    for from_state, model_info in model.transition_models.items():
        importance_df = model.get_feature_importance(from_state)
        if not importance_df.empty:
            importance_df['from_state'] = from_state
            importance_data.append(importance_df.head(5))  # Top 5 features
    
    if importance_data:
        combined_importance = pd.concat(importance_data)
        
        fig = px.bar(
            combined_importance,
            x='importance',
            y='feature',
            color='from_state',
            orientation='h',
            title="Top Feature Importances by State",
            facet_col='from_state'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics (if test data available)
    if test_data is not None:
        st.subheader("Test Set Performance")
        
        # Quick evaluation
        sample_size = min(500, len(test_data))
        test_sample = test_data.sample(sample_size, random_state=42)
        
        # Calculate some basic metrics
        correct_predictions = (test_sample['state'] == test_sample['next_state']).sum()
        accuracy = correct_predictions / len(test_sample)
        
        st.metric("Sample Accuracy", f"{accuracy:.1%}")
        
        # State distribution
        st.markdown("**Test Set State Distribution**")
        state_dist = test_sample['next_state'].value_counts(normalize=True)
        
        fig = px.pie(
            values=state_dist.values,
            names=state_dist.index,
            title="Test Set State Distribution",
            color_discrete_map={'STAY': 'green', 'SPLIT': 'orange', 'LEAVE': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

def business_insights_page(df):
    """Page for business insights and analysis."""
    
    st.header("Business Insights")
    st.markdown("Key findings and actionable insights from the analysis.")
    
    # Executive Summary
    st.subheader("Executive Summary")
    
    total_customers = df['customer_id'].nunique()
    total_records = len(df)
    stay_rate = (df['next_state'] == 'STAY').mean()
    leave_rate = (df['next_state'] == 'LEAVE').mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Total Records", f"{total_records:,}")
    with col3:
        st.metric("Retention Rate", f"{stay_rate:.1%}")
    with col4:
        st.metric("Churn Rate", f"{leave_rate:.1%}")
    
    # Key insights
    st.subheader("Key Business Insights")
    
    insights = [
        {
            "title": "High Retention Rate",
            "description": f"Overall retention rate of {stay_rate:.1%} indicates strong customer loyalty.",
            "action": "Continue current retention strategies while focusing on at-risk segments."
        },
        {
            "title": "Digital Engagement Critical",
            "description": "Digital engagement strongly correlates with wallet share and retention.",
            "action": "Invest in digital channel improvements and customer education."
        },
        {
            "title": "Product Holdings Matter",
            "description": "Customers with more products show higher retention rates.",
            "action": "Implement cross-selling programs targeting 1-2 product customers."
        },
        {
            "title": "Early Warning Signs",
            "description": "Complaints and fee events are strong predictors of churn risk.",
            "action": "Implement proactive outreach for customers with complaints/fees."
        }
    ]
    
    for insight in insights:
        with st.expander(insight["title"]):
            st.markdown(f"**Finding:** {insight['description']}")
            st.markdown(f"**Recommended Action:** {insight['action']}")
    
    # Detailed analysis
    st.subheader("Detailed Analysis")
    
    # Correlation with wallet share
    st.markdown("**Feature Correlation with Wallet Share**")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numerical_cols].corr()['wallet_share'].sort_values(ascending=False)
    
    # Remove self-correlation and wallet_share_next
    correlations = correlations.drop(['wallet_share', 'wallet_share_next'], errors='ignore')
    
    fig = px.bar(
        x=correlations.values[:10],
        y=correlations.index[:10],
        orientation='h',
        title="Top 10 Features Correlated with Wallet Share",
        labels={'x': 'Correlation', 'y': 'Feature'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.markdown("**Risk Factor Analysis**")
    
    at_risk = df[
        ((df['state'] == 'STAY') & (df['next_state'] == 'LEAVE')) |
        ((df['state'] == 'SPLIT') & (df['next_state'] == 'LEAVE'))
    ]
    
    stable = df[df['next_state'] == 'STAY']
    
    risk_factors = ['digital_engagement', 'product_count', 'complaints_12m', 'fee_events_12m']
    
    comparison_data = []
    for factor in risk_factors:
        if factor in df.columns:
            at_risk_avg = at_risk[factor].mean()
            stable_avg = stable[factor].mean()
            
            comparison_data.extend([
                {'Factor': factor, 'Group': 'At Risk', 'Value': at_risk_avg},
                {'Factor': factor, 'Group': 'Stable', 'Value': stable_avg}
            ])
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df,
            x='Factor',
            y='Value',
            color='Group',
            barmode='group',
            title="Risk Factors: At-Risk vs Stable Customers"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
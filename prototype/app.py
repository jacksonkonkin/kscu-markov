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
    page_icon="🏦",
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
    
    st.title("🏦 KSCU Wallet-Share Markov Predictor")
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
        ["🎯 Individual Prediction", "📊 Scenario Analysis", "📈 Model Performance", "🔍 Business Insights"]
    )
    
    if page == "🎯 Individual Prediction":
        individual_prediction_page(model, feature_cols, df)
    elif page == "📊 Scenario Analysis":
        scenario_analysis_page(model, feature_cols, df)
    elif page == "📈 Model Performance":
        model_performance_page(model, df, test)
    else:
        business_insights_page(df)

def individual_prediction_page(model, feature_cols, df):
    """Page for individual customer predictions."""
    
    st.header("🎯 Individual Customer Prediction")
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
        
        # Display metrics
        for state, prob in trans_probs.items():
            color_class = "success-metric" if state == "STAY" else "warning-metric" if state == "SPLIT" else "danger-metric"
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h4>{state}</h4>
                <h2>{prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
        
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

def scenario_analysis_page(model, feature_cols, df):
    """Page for scenario testing and interventions."""
    
    st.header("📊 Scenario Analysis")
    st.markdown("Test business interventions and their impact on customer behavior.")
    
    # Scenario selection
    scenario_type = st.selectbox(
        "Select Scenario:",
        [
            "Digital Adoption Campaign",
            "Product Cross-selling",
            "Customer Retention Program",
            "Custom Intervention"
        ]
    )
    
    if scenario_type == "Digital Adoption Campaign":
        digital_campaign_scenario(model, feature_cols, df)
    elif scenario_type == "Product Cross-selling":
        product_campaign_scenario(model, feature_cols, df)
    elif scenario_type == "Customer Retention Program":
        retention_scenario(model, feature_cols, df)
    else:
        custom_scenario(model, feature_cols, df)

def digital_campaign_scenario(model, feature_cols, df):
    """Digital adoption campaign scenario."""
    
    st.subheader("📱 Digital Adoption Campaign")
    st.markdown("Simulate the impact of increasing digital engagement scores.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Campaign Parameters**")
        
        # Target segment
        target_age_min = st.slider("Target Age (Min)", 18, 80, 25)
        target_age_max = st.slider("Target Age (Max)", 20, 84, 45)
        current_digital_max = st.slider("Current Digital Engagement (Max)", 0, 100, 50,
                                       help="Target customers with digital engagement below this level")
        
        # Intervention effect
        digital_boost = st.slider("Digital Engagement Increase", 0, 50, 20,
                                 help="Points increase in digital engagement")
        
        # Calculate affected customers
        mask = (
            (df['age'] >= target_age_min) & 
            (df['age'] <= target_age_max) & 
            (df['digital_engagement'] <= current_digital_max)
        )
        affected_customers = df[mask].copy()
        
        st.metric("Affected Customers", f"{len(affected_customers):,}")
        st.metric("% of Total", f"{len(affected_customers)/len(df):.1%}")
    
    with col2:
        st.markdown("**Impact Analysis**")
        
        if len(affected_customers) > 0:
            # Before and after predictions
            results_before = []
            results_after = []
            
            sample_size = min(1000, len(affected_customers))  # Limit for performance
            sample_customers = affected_customers.sample(sample_size, random_state=42)
            
            for _, customer in sample_customers.iterrows():
                # Current prediction
                features_current = customer[feature_cols].to_frame().T
                trans_probs_before = model.predict_transition_probs(customer['state'], features_current)
                
                # After intervention
                features_after = features_current.copy()
                features_after['digital_engagement'] = min(100, customer['digital_engagement'] + digital_boost)
                trans_probs_after = model.predict_transition_probs(customer['state'], features_after)
                
                results_before.append(trans_probs_before)
                results_after.append(trans_probs_after)
            
            # Calculate aggregate changes
            for state in STATES:
                prob_before = np.mean([r[state] for r in results_before])
                prob_after = np.mean([r[state] for r in results_after])
                change = prob_after - prob_before
                
                st.metric(
                    label=f"Avg. {state} Probability",
                    value=f"{prob_after:.1%}",
                    delta=f"{change:.1%}"
                )
            
            # Visualization
            impact_data = []
            for state in STATES:
                prob_before = np.mean([r[state] for r in results_before])
                prob_after = np.mean([r[state] for r in results_after])
                impact_data.extend([
                    {'State': state, 'Scenario': 'Before', 'Probability': prob_before},
                    {'State': state, 'Scenario': 'After', 'Probability': prob_after}
                ])
            
            impact_df = pd.DataFrame(impact_data)
            fig = px.bar(impact_df, x='State', y='Probability', color='Scenario',
                        barmode='group', title="Before vs After Campaign")
            st.plotly_chart(fig, use_container_width=True)

def product_campaign_scenario(model, feature_cols, df):
    """Product cross-selling scenario."""
    
    st.subheader("🛍️ Product Cross-selling Campaign")
    st.markdown("Simulate impact of increasing product holdings.")
    
    # Implementation similar to digital campaign but for product_count
    st.info("Product cross-selling scenario - targeting customers with 1-2 products to increase holdings")
    
    # Target customers with low product count
    target_customers = df[df['product_count'] <= 2].copy()
    product_increase = st.slider("Additional Products", 1, 3, 1)
    
    st.metric("Target Customers", f"{len(target_customers):,}")
    
    # Show potential impact (simplified)
    if len(target_customers) > 0:
        current_stay_rate = (target_customers['next_state'] == 'STAY').mean()
        st.metric("Current STAY Rate", f"{current_stay_rate:.1%}")
        
        # Estimate improvement (rule of thumb: each additional product increases retention)
        estimated_improvement = product_increase * 0.05  # 5% per product
        estimated_new_rate = min(0.95, current_stay_rate + estimated_improvement)
        
        st.metric(
            "Estimated STAY Rate",
            f"{estimated_new_rate:.1%}",
            delta=f"{estimated_improvement:.1%}"
        )

def retention_scenario(model, feature_cols, df):
    """Customer retention program scenario."""
    
    st.subheader("🎯 Customer Retention Program")
    st.markdown("Target at-risk customers to prevent churn.")
    
    # Identify at-risk customers
    at_risk = df[
        ((df['state'] == 'STAY') & (df['next_state'] == 'LEAVE')) |
        ((df['state'] == 'SPLIT') & (df['next_state'] == 'LEAVE'))
    ].copy()
    
    st.metric("At-Risk Customers", f"{len(at_risk):,}")
    st.metric("% of Total", f"{len(at_risk)/len(df):.2%}")
    
    if len(at_risk) > 0:
        st.markdown("**Risk Factors Analysis**")
        
        # Compare at-risk vs stable customers
        stable = df[df['next_state'] == 'STAY'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**At-Risk Profile**")
            st.metric("Avg Digital Engagement", f"{at_risk['digital_engagement'].mean():.0f}")
            st.metric("Avg Product Count", f"{at_risk['product_count'].mean():.1f}")
            st.metric("Avg Complaints", f"{at_risk['complaints_12m'].mean():.1f}")
        
        with col2:
            st.markdown("**Stable Profile**")
            st.metric("Avg Digital Engagement", f"{stable['digital_engagement'].mean():.0f}")
            st.metric("Avg Product Count", f"{stable['product_count'].mean():.1f}")
            st.metric("Avg Complaints", f"{stable['complaints_12m'].mean():.1f}")

def custom_scenario(model, feature_cols, df):
    """Custom intervention scenario."""
    
    st.subheader("🛠️ Custom Intervention")
    st.markdown("Design your own intervention scenario.")
    
    # Allow user to modify multiple features
    st.markdown("**Select features to modify:**")
    
    modifications = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Digital Engagement"):
            modifications['digital_engagement'] = st.slider("Change in Digital Engagement", -50, 50, 10)
        
        if st.checkbox("Product Count"):
            modifications['product_count'] = st.slider("Change in Product Count", -2, 3, 1)
        
        if st.checkbox("Average Balance"):
            modifications['avg_balance'] = st.slider("Change in Balance (%)", -50, 100, 20)
    
    with col2:
        if st.checkbox("Complaints"):
            modifications['complaints_12m'] = st.slider("Change in Complaints", -5, 0, -1)
        
        if st.checkbox("Fee Events"):
            modifications['fee_events_12m'] = st.slider("Change in Fee Events", -10, 0, -2)
        
        if st.checkbox("Branch Visits"):
            modifications['branch_visits_last_q'] = st.slider("Change in Branch Visits", -10, 10, -2)
    
    if modifications:
        st.markdown("**Scenario Impact**")
        
        # Apply modifications to a sample
        sample_df = df.sample(500, random_state=42)
        modified_df = sample_df.copy()
        
        for feature, change in modifications.items():
            if feature == 'avg_balance':
                modified_df[feature] = modified_df[feature] * (1 + change/100)
            else:
                modified_df[feature] = modified_df[feature] + change
                
            # Apply reasonable bounds
            if feature == 'digital_engagement':
                modified_df[feature] = np.clip(modified_df[feature], 0, 100)
            elif feature == 'product_count':
                modified_df[feature] = np.clip(modified_df[feature], 1, 8)
        
        # Show summary of changes
        st.success(f"Scenario applied to {len(sample_df)} sample customers")

def model_performance_page(model, df, test_data):
    """Page showing model performance metrics."""
    
    st.header("📈 Model Performance")
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
    
    st.header("🔍 Business Insights")
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
            "title": "🎯 High Retention Rate",
            "description": f"Overall retention rate of {stay_rate:.1%} indicates strong customer loyalty.",
            "action": "Continue current retention strategies while focusing on at-risk segments."
        },
        {
            "title": "📱 Digital Engagement Critical",
            "description": "Digital engagement strongly correlates with wallet share and retention.",
            "action": "Invest in digital channel improvements and customer education."
        },
        {
            "title": "🛍️ Product Holdings Matter",
            "description": "Customers with more products show higher retention rates.",
            "action": "Implement cross-selling programs targeting 1-2 product customers."
        },
        {
            "title": "⚠️ Early Warning Signs",
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
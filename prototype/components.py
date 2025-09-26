"""UI components for the KSCU Streamlit prototype."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional


def customer_input_form(defaults: Optional[Dict] = None) -> Dict:
    """
    Create a customer input form with default values.
    
    Args:
        defaults: Default values for the form fields
        
    Returns:
        Dictionary of customer features
    """
    if defaults is None:
        defaults = {
            'age': 45,
            'tenure_years': 5.0,
            'product_count': 3,
            'avg_balance': 25000,
            'has_mortgage': False,
            'salary_deposit_flag': True,
            'digital_engagement': 50,
            'branch_visits_last_q': 2,
            'complaints_12m': 0,
            'fee_events_12m': 2,
            'rate_sensitivity': 5.0,
            'card_spend_monthly': 1000,
            'nps_bucket': 7,
            'promo_exposure': 3
        }
    
    st.subheader("Customer Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 18, 84, defaults['age'])
        tenure = st.slider("Tenure (years)", 0.0, 20.0, defaults['tenure_years'], step=0.5)
        
        st.markdown("**Financial**")
        product_count = st.slider("Number of Products", 1, 8, defaults['product_count'])
        avg_balance = st.number_input("Average Balance ($)", 0, 500000, defaults['avg_balance'], step=1000)
        has_mortgage = st.checkbox("Has Mortgage", defaults['has_mortgage'])
        salary_deposit = st.checkbox("Salary Deposit", defaults['salary_deposit_flag'])
        
        st.markdown("**Spending**")
        card_spend = st.number_input("Monthly Card Spend ($)", 0, 20000, defaults['card_spend_monthly'], step=100)
        rate_sensitivity = st.slider("Rate Sensitivity", 0.0, 10.0, defaults['rate_sensitivity'], step=0.1)
    
    with col2:
        st.markdown("**Engagement**")
        digital_engagement = st.slider("Digital Engagement", 0, 100, defaults['digital_engagement'])
        branch_visits = st.slider("Branch Visits (last quarter)", 0, 30, defaults['branch_visits_last_q'])
        
        st.markdown("**Experience**")
        complaints = st.slider("Complaints (12m)", 0, 15, defaults['complaints_12m'])
        fee_events = st.slider("Fee Events (12m)", 0, 30, defaults['fee_events_12m'])
        
        st.markdown("**Marketing**")
        nps_bucket = st.slider("NPS Bucket", 1, 10, defaults['nps_bucket'])
        promo_exposure = st.slider("Promo Exposure", 0, 15, defaults['promo_exposure'])
    
    return {
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


def display_prediction_results(transition_probs: Dict, wallet_shares: Dict, current_state: str):
    """
    Display prediction results in a formatted layout.
    
    Args:
        transition_probs: Dictionary of transition probabilities
        wallet_shares: Dictionary of wallet share predictions by state
        current_state: Current customer state
    """
    st.subheader(f"Predictions for {current_state} Customer")
    
    # Transition probabilities
    st.markdown("### Next Quarter Transition Probabilities")
    
    # Create columns for each state with consistent ordering
    cols = st.columns(3)
    state_order = ["STAY", "SPLIT", "LEAVE"]  # Define order for consistency

    for i, state in enumerate(state_order):
        if state in transition_probs:
            prob = transition_probs[state]
            with cols[i]:
                # Enhanced color coding and styling
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
    
    # Wallet share predictions
    st.markdown("### Predicted Wallet Share by State")
    
    wallet_df = pd.DataFrame(list(wallet_shares.items()), columns=['State', 'Wallet_Share'])
    
    fig = px.bar(
        wallet_df, 
        x='State', 
        y='Wallet_Share',
        color='State',
        color_discrete_map={'STAY': 'green', 'SPLIT': 'orange', 'LEAVE': 'red'},
        title="Predicted Wallet Share by State",
        labels={'Wallet_Share': 'Wallet Share'}
    )
    fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Expected value calculation
    expected_wallet_share = sum(transition_probs[state] * wallet_shares[state] for state in transition_probs.keys())
    
    st.markdown("### Summary Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Most Likely State", max(transition_probs, key=transition_probs.get))
    
    with col2:
        st.metric("Expected Wallet Share", f"{expected_wallet_share:.1%}")
    
    with col3:
        churn_risk = transition_probs.get('LEAVE', 0)
        risk_level = "High" if churn_risk > 0.2 else "Medium" if churn_risk > 0.1 else "Low"
        st.metric("Churn Risk", f"{risk_level} ({churn_risk:.1%})")


def create_scenario_comparison_chart(before_data: Dict, after_data: Dict, title: str = "Scenario Comparison"):
    """
    Create a comparison chart for before/after scenario analysis.
    
    Args:
        before_data: Dictionary of metrics before intervention
        after_data: Dictionary of metrics after intervention
        title: Chart title
    """
    comparison_data = []
    
    for metric, value_before in before_data.items():
        value_after = after_data.get(metric, value_before)
        
        comparison_data.extend([
            {'Metric': metric, 'Scenario': 'Before', 'Value': value_before},
            {'Metric': metric, 'Scenario': 'After', 'Value': value_after}
        ])
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df,
        x='Metric',
        y='Value',
        color='Scenario',
        barmode='group',
        title=title
    )
    
    return fig


def display_risk_assessment(customer_features: Dict, risk_factors: List[str]):
    """
    Display risk assessment for a customer.
    
    Args:
        customer_features: Dictionary of customer features
        risk_factors: List of features that indicate risk
    """
    st.subheader("Risk Assessment")
    
    risk_score = 0
    risk_details = []
    
    # Define risk thresholds (these could be data-driven)
    risk_thresholds = {
        'digital_engagement': {'low': 30, 'medium': 60},
        'product_count': {'low': 2, 'medium': 3},
        'complaints_12m': {'low': 0, 'medium': 2},
        'fee_events_12m': {'low': 1, 'medium': 5},
        'branch_visits_last_q': {'low': 5, 'medium': 10}
    }
    
    for factor in risk_factors:
        if factor in customer_features and factor in risk_thresholds:
            value = customer_features[factor]
            thresholds = risk_thresholds[factor]
            
            if factor in ['digital_engagement', 'product_count']:
                # Lower values = higher risk
                if value <= thresholds['low']:
                    risk_level = "High"
                    risk_score += 3
                elif value <= thresholds['medium']:
                    risk_level = "Medium"
                    risk_score += 2
                else:
                    risk_level = "Low"
                    risk_score += 1
            else:
                # Higher values = higher risk
                if value >= thresholds['medium']:
                    risk_level = "High"
                    risk_score += 3
                elif value >= thresholds['low']:
                    risk_level = "Medium"
                    risk_score += 2
                else:
                    risk_level = "Low"
                    risk_score += 1
            
            risk_details.append({
                'Factor': factor.replace('_', ' ').title(),
                'Value': value,
                'Risk Level': risk_level
            })
    
    # Display overall risk score
    max_possible_score = len(risk_factors) * 3
    risk_percentage = (risk_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    if risk_percentage >= 70:
        overall_risk = "HIGH RISK"
        color = "red"
    elif risk_percentage >= 40:
        overall_risk = "MEDIUM RISK"
        color = "orange"
    else:
        overall_risk = "LOW RISK"
        color = "green"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; border: 2px solid {color}; border-radius: 0.5rem; background-color: {'#ffebee' if color == 'red' else '#fff3e0' if color == 'orange' else '#e8f5e8'};">
        <h2 style="color: {color}; margin: 0;">{overall_risk}</h2>
        <p style="margin: 0.5rem 0;">Risk Score: {risk_score}/{max_possible_score} ({risk_percentage:.0f}%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display risk factor details
    if risk_details:
        st.markdown("### Risk Factor Breakdown")
        risk_df = pd.DataFrame(risk_details)
        
        # Color code the risk levels
        def color_risk_level(val):
            if val == 'High':
                return 'background-color: #ffcdd2'
            elif val == 'Medium':
                return 'background-color: #ffe0b2'
            else:
                return 'background-color: #c8e6c9'
        
        styled_df = risk_df.style.applymap(color_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True)


def create_feature_importance_chart(feature_importance: pd.DataFrame, title: str = "Feature Importance"):
    """
    Create a feature importance visualization.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        title: Chart title
    """
    if feature_importance.empty:
        st.info("No feature importance data available.")
        return None
    
    # Take top 10 features
    top_features = feature_importance.head(10)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=title,
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    return fig


def display_model_metrics(metrics: Dict):
    """
    Display model performance metrics in a formatted layout.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    st.subheader("Model Performance Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.1%}")
    
    with col2:
        log_loss = metrics.get('log_loss', 0)
        st.metric("Log Loss", f"{log_loss:.3f}")
    
    with col3:
        wallet_mae = metrics.get('wallet_mae', 0)
        st.metric("Wallet Share MAE", f"{wallet_mae:.4f}")
    
    with col4:
        f1_weighted = metrics.get('f1_weighted', 0)
        st.metric("F1 (Weighted)", f"{f1_weighted:.3f}")
    
    # Per-class metrics
    if any(key.startswith('f1_') for key in metrics.keys()):
        st.markdown("### Per-Class Performance")
        
        states = ['STAY', 'SPLIT', 'LEAVE']
        class_metrics = []
        
        for state in states:
            precision = metrics.get(f'precision_{state}', 0)
            recall = metrics.get(f'recall_{state}', 0)
            f1 = metrics.get(f'f1_{state}', 0)
            
            class_metrics.append({
                'State': state,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        
        class_df = pd.DataFrame(class_metrics)
        
        # Create grouped bar chart
        fig = px.bar(
            class_df.melt(id_vars=['State'], var_name='Metric', value_name='Score'),
            x='State',
            y='Score',
            color='Metric',
            barmode='group',
            title="Per-Class Performance Metrics"
        )
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)


def create_transition_matrix_heatmap(transition_matrix: pd.DataFrame, title: str = "Transition Matrix"):
    """
    Create a heatmap visualization of the transition matrix.
    
    Args:
        transition_matrix: Transition probability matrix
        title: Chart title
    """
    # Convert to percentage
    trans_matrix_pct = transition_matrix * 100
    
    fig = px.imshow(
        trans_matrix_pct.values,
        labels=dict(x="Next State", y="Current State", color="Probability (%)"),
        x=trans_matrix_pct.columns,
        y=trans_matrix_pct.index,
        color_continuous_scale="RdYlGn",
        title=title,
        text_auto=".1f"
    )
    
    fig.update_traces(texttemplate="%{text}%", textfont_size=12)
    
    return fig


def scenario_selector():
    """
    Create a scenario selection interface.
    
    Returns:
        Dictionary with scenario configuration
    """
    st.subheader("Select Scenario")
    
    scenario_type = st.selectbox(
        "Intervention Type:",
        [
            "Digital Adoption Campaign",
            "Product Cross-selling",
            "Fee Reduction Program",
            "Customer Service Improvement",
            "Custom Intervention"
        ]
    )
    
    scenario_config = {'type': scenario_type}
    
    if scenario_type == "Digital Adoption Campaign":
        scenario_config.update({
            'target_engagement_max': st.slider("Target customers with digital engagement below:", 0, 100, 50),
            'engagement_boost': st.slider("Digital engagement increase (points):", 0, 50, 20),
            'target_age_min': st.slider("Minimum age:", 18, 80, 25),
            'target_age_max': st.slider("Maximum age:", 20, 84, 55)
        })
    
    elif scenario_type == "Product Cross-selling":
        scenario_config.update({
            'target_products_max': st.slider("Target customers with products ≤:", 1, 5, 2),
            'additional_products': st.slider("Additional products to sell:", 1, 3, 1),
            'target_balance_min': st.number_input("Minimum balance ($):", 0, 100000, 10000, step=5000)
        })
    
    elif scenario_type == "Fee Reduction Program":
        scenario_config.update({
            'target_fee_events_min': st.slider("Target customers with fee events ≥:", 1, 20, 5),
            'fee_reduction': st.slider("Reduction in fee events:", 1, 10, 3)
        })
    
    elif scenario_type == "Customer Service Improvement":
        scenario_config.update({
            'target_complaints_min': st.slider("Target customers with complaints ≥:", 1, 10, 2),
            'complaint_reduction': st.slider("Reduction in complaints:", 1, 5, 2)
        })
    
    else:  # Custom Intervention
        st.markdown("**Select features to modify:**")
        
        modifications = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Digital Engagement"):
                modifications['digital_engagement'] = st.slider("Change in Digital Engagement:", -50, 50, 10)
            
            if st.checkbox("Product Count"):
                modifications['product_count'] = st.slider("Change in Products:", -2, 3, 1)
            
            if st.checkbox("Average Balance"):
                modifications['avg_balance'] = st.slider("Change in Balance (%):", -50, 100, 20)
        
        with col2:
            if st.checkbox("Complaints"):
                modifications['complaints_12m'] = st.slider("Change in Complaints:", -5, 0, -1)
            
            if st.checkbox("Fee Events"):
                modifications['fee_events_12m'] = st.slider("Change in Fee Events:", -10, 0, -2)
            
            if st.checkbox("Branch Visits"):
                modifications['branch_visits_last_q'] = st.slider("Change in Branch Visits:", -10, 10, -2)
        
        scenario_config['modifications'] = modifications
    
    return scenario_config
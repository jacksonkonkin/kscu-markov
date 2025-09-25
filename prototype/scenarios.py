"""Scenario testing logic for the KSCU prototype."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class ScenarioEngine:
    """Engine for running business scenario simulations."""
    
    def __init__(self, model, feature_cols: List[str]):
        """
        Initialize the scenario engine.
        
        Args:
            model: Trained Markov model
            feature_cols: List of feature columns
        """
        self.model = model
        self.feature_cols = feature_cols
    
    def run_digital_campaign(self, 
                           df: pd.DataFrame,
                           target_age_range: Tuple[int, int] = (25, 55),
                           current_engagement_max: int = 50,
                           engagement_boost: int = 20,
                           sample_size: int = 1000) -> Dict:
        """
        Simulate a digital adoption campaign.
        
        Args:
            df: Customer dataset
            target_age_range: (min_age, max_age) for targeting
            current_engagement_max: Target customers below this engagement level
            engagement_boost: Points to increase digital engagement
            sample_size: Number of customers to sample for simulation
            
        Returns:
            Dictionary with campaign results
        """
        
        # Identify target customers
        mask = (
            (df['age'] >= target_age_range[0]) & 
            (df['age'] <= target_age_range[1]) & 
            (df['digital_engagement'] <= current_engagement_max)
        )
        
        target_customers = df[mask].copy()
        
        if len(target_customers) == 0:
            return {
                'error': 'No customers match the targeting criteria',
                'affected_customers': 0
            }
        
        # Sample for performance
        if len(target_customers) > sample_size:
            sample_customers = target_customers.sample(sample_size, random_state=42)
        else:
            sample_customers = target_customers
        
        # Run before/after simulation
        before_results = self._predict_batch(sample_customers)
        
        # Apply intervention
        modified_customers = sample_customers.copy()
        modified_customers['digital_engagement'] = np.clip(
            modified_customers['digital_engagement'] + engagement_boost, 0, 100
        )
        
        after_results = self._predict_batch(modified_customers)
        
        # Calculate impact
        impact = self._calculate_impact(before_results, after_results)
        
        return {
            'campaign_type': 'Digital Adoption',
            'target_criteria': f"Age {target_age_range[0]}-{target_age_range[1]}, Digital Engagement ≤ {current_engagement_max}",
            'intervention': f"+{engagement_boost} digital engagement points",
            'total_eligible': len(target_customers),
            'simulated_customers': len(sample_customers),
            'eligibility_rate': len(target_customers) / len(df),
            'before_results': before_results,
            'after_results': after_results,
            'impact': impact
        }
    
    def run_product_campaign(self,
                           df: pd.DataFrame,
                           target_products_max: int = 2,
                           additional_products: int = 1,
                           min_balance: float = 10000,
                           sample_size: int = 1000) -> Dict:
        """
        Simulate a product cross-selling campaign.
        
        Args:
            df: Customer dataset
            target_products_max: Target customers with ≤ this many products
            additional_products: Number of additional products to sell
            min_balance: Minimum balance for targeting
            sample_size: Number of customers to sample
            
        Returns:
            Dictionary with campaign results
        """
        
        # Identify target customers
        mask = (
            (df['product_count'] <= target_products_max) &
            (df['avg_balance'] >= min_balance)
        )
        
        target_customers = df[mask].copy()
        
        if len(target_customers) == 0:
            return {
                'error': 'No customers match the targeting criteria',
                'affected_customers': 0
            }
        
        # Sample for performance
        if len(target_customers) > sample_size:
            sample_customers = target_customers.sample(sample_size, random_state=42)
        else:
            sample_customers = target_customers
        
        # Run before/after simulation
        before_results = self._predict_batch(sample_customers)
        
        # Apply intervention
        modified_customers = sample_customers.copy()
        modified_customers['product_count'] = np.clip(
            modified_customers['product_count'] + additional_products, 1, 8
        )
        
        after_results = self._predict_batch(modified_customers)
        
        # Calculate impact
        impact = self._calculate_impact(before_results, after_results)
        
        return {
            'campaign_type': 'Product Cross-selling',
            'target_criteria': f"Products ≤ {target_products_max}, Balance ≥ ${min_balance:,}",
            'intervention': f"+{additional_products} additional products",
            'total_eligible': len(target_customers),
            'simulated_customers': len(sample_customers),
            'eligibility_rate': len(target_customers) / len(df),
            'before_results': before_results,
            'after_results': after_results,
            'impact': impact
        }
    
    def run_fee_reduction(self,
                         df: pd.DataFrame,
                         target_fee_events_min: int = 5,
                         fee_reduction: int = 3,
                         sample_size: int = 1000) -> Dict:
        """
        Simulate a fee reduction program.
        
        Args:
            df: Customer dataset
            target_fee_events_min: Target customers with ≥ this many fee events
            fee_reduction: Number of fee events to reduce
            sample_size: Number of customers to sample
            
        Returns:
            Dictionary with campaign results
        """
        
        # Identify target customers
        mask = df['fee_events_12m'] >= target_fee_events_min
        target_customers = df[mask].copy()
        
        if len(target_customers) == 0:
            return {
                'error': 'No customers match the targeting criteria',
                'affected_customers': 0
            }
        
        # Sample for performance
        if len(target_customers) > sample_size:
            sample_customers = target_customers.sample(sample_size, random_state=42)
        else:
            sample_customers = target_customers
        
        # Run before/after simulation
        before_results = self._predict_batch(sample_customers)
        
        # Apply intervention
        modified_customers = sample_customers.copy()
        modified_customers['fee_events_12m'] = np.maximum(
            modified_customers['fee_events_12m'] - fee_reduction, 0
        )
        
        after_results = self._predict_batch(modified_customers)
        
        # Calculate impact
        impact = self._calculate_impact(before_results, after_results)
        
        return {
            'campaign_type': 'Fee Reduction',
            'target_criteria': f"Fee Events ≥ {target_fee_events_min}",
            'intervention': f"-{fee_reduction} fee events",
            'total_eligible': len(target_customers),
            'simulated_customers': len(sample_customers),
            'eligibility_rate': len(target_customers) / len(df),
            'before_results': before_results,
            'after_results': after_results,
            'impact': impact
        }
    
    def run_service_improvement(self,
                              df: pd.DataFrame,
                              target_complaints_min: int = 2,
                              complaint_reduction: int = 2,
                              sample_size: int = 1000) -> Dict:
        """
        Simulate a customer service improvement program.
        
        Args:
            df: Customer dataset
            target_complaints_min: Target customers with ≥ this many complaints
            complaint_reduction: Number of complaints to reduce
            sample_size: Number of customers to sample
            
        Returns:
            Dictionary with campaign results
        """
        
        # Identify target customers
        mask = df['complaints_12m'] >= target_complaints_min
        target_customers = df[mask].copy()
        
        if len(target_customers) == 0:
            return {
                'error': 'No customers match the targeting criteria',
                'affected_customers': 0
            }
        
        # Sample for performance
        if len(target_customers) > sample_size:
            sample_customers = target_customers.sample(sample_size, random_state=42)
        else:
            sample_customers = target_customers
        
        # Run before/after simulation
        before_results = self._predict_batch(sample_customers)
        
        # Apply intervention
        modified_customers = sample_customers.copy()
        modified_customers['complaints_12m'] = np.maximum(
            modified_customers['complaints_12m'] - complaint_reduction, 0
        )
        
        after_results = self._predict_batch(modified_customers)
        
        # Calculate impact
        impact = self._calculate_impact(before_results, after_results)
        
        return {
            'campaign_type': 'Service Improvement',
            'target_criteria': f"Complaints ≥ {target_complaints_min}",
            'intervention': f"-{complaint_reduction} complaints",
            'total_eligible': len(target_customers),
            'simulated_customers': len(sample_customers),
            'eligibility_rate': len(target_customers) / len(df),
            'before_results': before_results,
            'after_results': after_results,
            'impact': impact
        }
    
    def run_custom_intervention(self,
                              df: pd.DataFrame,
                              modifications: Dict[str, float],
                              target_filter: Optional[Dict] = None,
                              sample_size: int = 1000) -> Dict:
        """
        Run a custom intervention scenario.
        
        Args:
            df: Customer dataset
            modifications: Dictionary of feature modifications
            target_filter: Optional filter criteria
            sample_size: Number of customers to sample
            
        Returns:
            Dictionary with intervention results
        """
        
        # Apply target filter if provided
        if target_filter:
            mask = pd.Series(True, index=df.index)
            for feature, (operator, value) in target_filter.items():
                if feature in df.columns:
                    if operator == '>=':
                        mask &= (df[feature] >= value)
                    elif operator == '<=':
                        mask &= (df[feature] <= value)
                    elif operator == '==':
                        mask &= (df[feature] == value)
            
            target_customers = df[mask].copy()
        else:
            target_customers = df.copy()
        
        if len(target_customers) == 0:
            return {
                'error': 'No customers match the targeting criteria',
                'affected_customers': 0
            }
        
        # Sample for performance
        if len(target_customers) > sample_size:
            sample_customers = target_customers.sample(sample_size, random_state=42)
        else:
            sample_customers = target_customers
        
        # Run before/after simulation
        before_results = self._predict_batch(sample_customers)
        
        # Apply modifications
        modified_customers = sample_customers.copy()
        
        for feature, change in modifications.items():
            if feature in modified_customers.columns:
                if feature == 'avg_balance':
                    # Percentage change for balance
                    modified_customers[feature] = modified_customers[feature] * (1 + change/100)
                else:
                    # Absolute change for other features
                    modified_customers[feature] = modified_customers[feature] + change
                
                # Apply bounds
                if feature == 'digital_engagement':
                    modified_customers[feature] = np.clip(modified_customers[feature], 0, 100)
                elif feature == 'product_count':
                    modified_customers[feature] = np.clip(modified_customers[feature], 1, 8)
                elif feature in ['complaints_12m', 'fee_events_12m', 'branch_visits_last_q']:
                    modified_customers[feature] = np.maximum(modified_customers[feature], 0)
        
        after_results = self._predict_batch(modified_customers)
        
        # Calculate impact
        impact = self._calculate_impact(before_results, after_results)
        
        return {
            'campaign_type': 'Custom Intervention',
            'target_criteria': target_filter or 'All customers',
            'intervention': modifications,
            'total_eligible': len(target_customers),
            'simulated_customers': len(sample_customers),
            'eligibility_rate': len(target_customers) / len(df),
            'before_results': before_results,
            'after_results': after_results,
            'impact': impact
        }
    
    def _predict_batch(self, customers_df: pd.DataFrame) -> Dict:
        """
        Run predictions for a batch of customers.
        
        Args:
            customers_df: DataFrame of customers
            
        Returns:
            Dictionary with prediction results
        """
        state_predictions = []
        state_probabilities = {state: [] for state in self.model.states}
        wallet_predictions = []
        
        for _, customer in customers_df.iterrows():
            # Prepare features
            features = customer[self.feature_cols].to_frame().T
            
            # Get state transition probabilities
            trans_probs = self.model.predict_transition_probs(customer['state'], features)
            
            # Store probabilities
            for state in self.model.states:
                state_probabilities[state].append(trans_probs.get(state, 0))
            
            # Predict most likely state
            pred_state = max(trans_probs, key=trans_probs.get)
            state_predictions.append(pred_state)
            
            # Predict wallet share
            wallet_pred = self.model.predict_wallet_share(pred_state, features)
            wallet_predictions.append(wallet_pred)
        
        # Calculate aggregate metrics
        results = {
            'state_probabilities': {
                state: np.mean(probs) for state, probs in state_probabilities.items()
            },
            'avg_wallet_share': np.mean(wallet_predictions),
            'stay_probability': state_probabilities['STAY'] and np.mean(state_probabilities['STAY']) or 0,
            'leave_probability': state_probabilities['LEAVE'] and np.mean(state_probabilities['LEAVE']) or 0,
            'split_probability': state_probabilities['SPLIT'] and np.mean(state_probabilities['SPLIT']) or 0,
            'predicted_states': state_predictions,
            'predicted_wallet_shares': wallet_predictions
        }
        
        return results
    
    def _calculate_impact(self, before_results: Dict, after_results: Dict) -> Dict:
        """
        Calculate the impact of an intervention.
        
        Args:
            before_results: Results before intervention
            after_results: Results after intervention
            
        Returns:
            Dictionary with impact metrics
        """
        impact = {}
        
        # State probability changes
        for state in ['STAY', 'SPLIT', 'LEAVE']:
            before_prob = before_results['state_probabilities'].get(state, 0)
            after_prob = after_results['state_probabilities'].get(state, 0)
            impact[f'{state.lower()}_change'] = after_prob - before_prob
            impact[f'{state.lower()}_change_pct'] = ((after_prob - before_prob) / before_prob * 100) if before_prob > 0 else 0
        
        # Wallet share change
        before_wallet = before_results['avg_wallet_share']
        after_wallet = after_results['avg_wallet_share']
        impact['wallet_share_change'] = after_wallet - before_wallet
        impact['wallet_share_change_pct'] = (after_wallet - before_wallet) / before_wallet * 100 if before_wallet > 0 else 0
        
        # Overall effectiveness score
        stay_improvement = impact['stay_change']
        leave_reduction = -impact['leave_change']  # Negative because we want to reduce LEAVE
        wallet_improvement = impact['wallet_share_change']
        
        # Weighted score (higher is better)
        effectiveness_score = (
            stay_improvement * 0.4 +
            leave_reduction * 0.4 +
            wallet_improvement * 0.2
        )
        
        impact['effectiveness_score'] = effectiveness_score
        
        # Risk assessment
        if impact['leave_change'] > 0.02:  # 2% increase in leave probability
            impact['risk_level'] = 'High Risk'
        elif impact['leave_change'] > 0.005:  # 0.5% increase
            impact['risk_level'] = 'Medium Risk'
        else:
            impact['risk_level'] = 'Low Risk'
        
        return impact


def calculate_roi_estimate(campaign_results: Dict, 
                          avg_customer_value: float = 2500,
                          campaign_cost_per_customer: float = 50) -> Dict:
    """
    Calculate estimated ROI for a campaign.
    
    Args:
        campaign_results: Results from scenario simulation
        avg_customer_value: Average annual value per customer
        campaign_cost_per_customer: Cost to target each customer
        
    Returns:
        Dictionary with ROI estimates
    """
    
    if 'error' in campaign_results:
        return {'error': campaign_results['error']}
    
    simulated_customers = campaign_results['simulated_customers']
    total_eligible = campaign_results['total_eligible']
    impact = campaign_results['impact']
    
    # Calculate value changes
    wallet_share_improvement = impact['wallet_share_change']
    leave_reduction = -impact['leave_change']  # Negative because we want to reduce leave
    
    # Estimate value per customer
    value_from_wallet_improvement = wallet_share_improvement * avg_customer_value
    value_from_retention = leave_reduction * avg_customer_value * 0.5  # Assume half the customer value for retention
    
    total_value_per_customer = value_from_wallet_improvement + value_from_retention
    
    # Calculate totals
    total_campaign_cost = total_eligible * campaign_cost_per_customer
    total_value = total_eligible * total_value_per_customer
    
    net_value = total_value - total_campaign_cost
    roi_ratio = (total_value / total_campaign_cost) if total_campaign_cost > 0 else 0
    
    return {
        'total_eligible_customers': total_eligible,
        'campaign_cost_per_customer': campaign_cost_per_customer,
        'total_campaign_cost': total_campaign_cost,
        'value_per_customer': total_value_per_customer,
        'total_value': total_value,
        'net_value': net_value,
        'roi_ratio': roi_ratio,
        'roi_percentage': (roi_ratio - 1) * 100 if roi_ratio > 0 else -100,
        'payback_achievable': net_value > 0,
        'value_breakdown': {
            'from_wallet_improvement': value_from_wallet_improvement,
            'from_retention': value_from_retention
        }
    }
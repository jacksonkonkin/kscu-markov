"""Markov chain model implementation for wallet share prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import STATES, MODEL_FEATURES, MARKOV_PARAMS
except ImportError:
    from config import STATES, MODEL_FEATURES, MARKOV_PARAMS


class MarkovChainModel:
    """Markov chain model with feature-dependent transitions."""
    
    def __init__(self, 
                 smoothing_alpha: float = 0.01,
                 use_features: bool = True):
        """
        Initialize Markov chain model.
        
        Args:
            smoothing_alpha: Laplace smoothing parameter
            use_features: Whether to use features for transition probabilities
        """
        self.states = STATES
        self.smoothing_alpha = smoothing_alpha
        self.use_features = use_features
        
        # Transition matrix (base)
        self.transition_matrix = None
        self.transition_counts = None
        
        # Feature-based models (one per state transition)
        self.transition_models = {}
        
        # Wallet share prediction models (one per state)
        self.wallet_share_models = {}
        
    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        """
        Fit the Markov chain model.
        
        Args:
            df: DataFrame with columns 'state', 'next_state', 'wallet_share', 'wallet_share_next'
            feature_cols: List of feature columns to use
        """
        if feature_cols is None:
            feature_cols = [col for col in MODEL_FEATURES if col in df.columns]
        
        # Calculate base transition matrix
        self._calculate_transition_matrix(df)
        
        if self.use_features and feature_cols:
            # Train feature-based transition models
            self._train_transition_models(df, feature_cols)
        
        # Train wallet share prediction models
        self._train_wallet_share_models(df, feature_cols)
        
    def _calculate_transition_matrix(self, df: pd.DataFrame):
        """Calculate base transition probability matrix."""
        # Count transitions
        self.transition_counts = pd.crosstab(df['state'], df['next_state'])
        
        # Add smoothing to avoid zero probabilities
        self.transition_counts = self.transition_counts + self.smoothing_alpha
        
        # Calculate probabilities (normalize by row)
        self.transition_matrix = self.transition_counts.div(
            self.transition_counts.sum(axis=1), axis=0
        )
        
        print("Base Transition Matrix:")
        print(self.transition_matrix.round(3))
        
    def _train_transition_models(self, df: pd.DataFrame, feature_cols: List[str]):
        """Train feature-dependent transition models."""
        
        for from_state in self.states:
            # Filter data for current state
            state_data = df[df['state'] == from_state].copy()
            
            if len(state_data) < 100:  # Skip if too few samples
                continue
            
            # Prepare features and target
            X = state_data[feature_cols].fillna(0)
            y = state_data['next_state']
            
            # Train model (using Random Forest for better feature interactions)
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            self.transition_models[from_state] = {
                'model': model,
                'features': feature_cols,
                'classes': model.classes_
            }
            
            print(f"Trained transition model for {from_state} -> ? (accuracy: {model.score(X, y):.3f})")
    
    def _train_wallet_share_models(self, df: pd.DataFrame, feature_cols: List[str]):
        """Train wallet share prediction models for each state."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        for state in self.states:
            # Filter data for current state
            state_data = df[df['next_state'] == state].copy()
            
            if len(state_data) < 50:  # Skip if too few samples
                continue
            
            # Prepare features and target
            X = state_data[feature_cols].fillna(0)
            y = state_data['wallet_share_next']
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            
            self.wallet_share_models[state] = {
                'model': model,
                'features': feature_cols,
                'mean': y.mean(),
                'std': y.std()
            }
            
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y, model.predict(X))
            print(f"Trained wallet share model for {state} (MAE: {mae:.4f})")
    
    def predict_transition_probs(self, current_state: str, features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Predict transition probabilities from current state.
        
        Args:
            current_state: Current state
            features: Feature values (optional)
            
        Returns:
            Dictionary of transition probabilities
        """
        if not self.use_features or features is None or current_state not in self.transition_models:
            # Use base transition matrix
            if current_state in self.transition_matrix.index:
                probs = self.transition_matrix.loc[current_state].to_dict()
            else:
                # Equal probabilities if state not seen
                probs = {state: 1.0 / len(self.states) for state in self.states}
        else:
            # Use feature-based model
            model_info = self.transition_models[current_state]
            model = model_info['model']
            
            # Ensure features are in correct order
            X = features[model_info['features']].fillna(0)
            
            # Get probability predictions
            prob_array = model.predict_proba(X)[0]
            probs = dict(zip(model_info['classes'], prob_array))
            
            # Ensure all states have probabilities
            for state in self.states:
                if state not in probs:
                    probs[state] = 0.0
        
        return probs
    
    def predict_wallet_share(self, next_state: str, features: Optional[pd.DataFrame] = None) -> float:
        """
        Predict wallet share for a given state.
        
        Args:
            next_state: The state to predict wallet share for
            features: Feature values (optional)
            
        Returns:
            Predicted wallet share
        """
        if next_state not in self.wallet_share_models or features is None:
            # Return state-specific average
            state_means = {'STAY': 0.9, 'SPLIT': 0.5, 'LEAVE': 0.1}
            return state_means.get(next_state, 0.5)
        
        model_info = self.wallet_share_models[next_state]
        model = model_info['model']
        
        # Prepare features
        X = features[model_info['features']].fillna(0)
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Clip to [0, 1] range
        return np.clip(prediction, 0, 1)
    
    def simulate_trajectory(self, 
                           initial_state: str,
                           features: pd.DataFrame,
                           n_steps: int = 4) -> Dict:
        """
        Simulate customer trajectory over multiple time steps.
        
        Args:
            initial_state: Starting state
            features: Initial feature values
            n_steps: Number of steps to simulate
            
        Returns:
            Dictionary with trajectory information
        """
        trajectory = {
            'states': [initial_state],
            'wallet_shares': [],
            'transition_probs': []
        }
        
        current_state = initial_state
        current_features = features.copy()
        
        for step in range(n_steps):
            # Get transition probabilities
            trans_probs = self.predict_transition_probs(current_state, current_features)
            trajectory['transition_probs'].append(trans_probs)
            
            # Sample next state
            states = list(trans_probs.keys())
            probs = list(trans_probs.values())
            next_state = np.random.choice(states, p=probs)
            
            # Predict wallet share
            wallet_share = self.predict_wallet_share(next_state, current_features)
            
            trajectory['states'].append(next_state)
            trajectory['wallet_shares'].append(wallet_share)
            
            # Update for next iteration
            current_state = next_state
            
            # Optionally update features based on state change
            # (simplified - in reality would need time-series model)
            if 'wallet_share' in current_features.columns:
                current_features['wallet_share'] = wallet_share
        
        return trajectory
    
    def get_feature_importance(self, from_state: str) -> pd.DataFrame:
        """Get feature importance for transitions from a given state."""
        if from_state not in self.transition_models:
            return pd.DataFrame()
        
        model_info = self.transition_models[from_state]
        model = model_info['model']
        features = model_info['features']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()


def calculate_steady_state(transition_matrix: pd.DataFrame, 
                          max_iterations: int = 1000,
                          tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Calculate steady-state distribution of Markov chain.
    
    Args:
        transition_matrix: Transition probability matrix
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary of steady-state probabilities
    """
    # Convert to numpy array
    P = transition_matrix.values
    n_states = len(P)
    
    # Initial distribution (uniform)
    pi = np.ones(n_states) / n_states
    
    # Power iteration
    for _ in range(max_iterations):
        pi_new = pi @ P
        
        if np.allclose(pi, pi_new, atol=tolerance):
            break
        
        pi = pi_new
    
    # Return as dictionary
    steady_state = dict(zip(transition_matrix.index, pi))
    return steady_state


if __name__ == "__main__":
    # Example usage
    from .preprocessing import load_raw_data, create_features
    
    # Load and prepare data
    df = load_raw_data()
    df = create_features(df)
    
    # Initialize and train model
    model = MarkovChainModel(use_features=True)
    
    feature_cols = [col for col in MODEL_FEATURES if col in df.columns]
    model.fit(df, feature_cols)
    
    # Calculate steady state
    steady_state = calculate_steady_state(model.transition_matrix)
    print("\nSteady State Distribution:")
    for state, prob in steady_state.items():
        print(f"  {state}: {prob:.3f}")
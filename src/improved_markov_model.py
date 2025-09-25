"""
Improved Markov Model with F1-LEAVE Optimization
Implements balanced approach: threshold optimization + cost-sensitive learning
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import STATES, MODEL_FEATURES, MARKOV_PARAMS
except ImportError:
    from config import STATES, MODEL_FEATURES, MARKOV_PARAMS


class ImprovedMarkovChainModel:
    """Enhanced Markov chain model with F1-LEAVE optimization."""

    def __init__(self,
                 smoothing_alpha: float = 0.01,
                 use_features: bool = True,
                 class_weights: Optional[Dict[str, float]] = None,
                 optimize_thresholds: bool = True):
        """
        Initialize improved Markov chain model.

        Args:
            smoothing_alpha: Laplace smoothing parameter
            use_features: Whether to use features for transition probabilities
            class_weights: Custom class weights for cost-sensitive learning
            optimize_thresholds: Whether to optimize classification thresholds
        """
        self.states = STATES
        self.smoothing_alpha = smoothing_alpha
        self.use_features = use_features
        self.optimize_thresholds = optimize_thresholds

        # Default balanced class weights (conservative approach)
        self.class_weights = class_weights or {
            'STAY': 1.0,     # Baseline
            'SPLIT': 2.0,    # Moderate boost
            'LEAVE': 3.0     # Conservative boost (not too aggressive)
        }

        # Model components
        self.transition_matrix = None
        self.transition_counts = None
        self.transition_models = {}
        self.wallet_share_models = {}

        # Threshold optimization
        self.optimal_thresholds = None
        self.validation_metrics = {}

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
            validation_data: Optional[pd.DataFrame] = None):
        """
        Fit the improved Markov chain model.

        Args:
            df: Training DataFrame
            feature_cols: List of feature columns to use
            validation_data: Validation set for threshold optimization
        """
        print(f"🤖 Training Improved Markov Model...")
        print(f"   Class weights: {self.class_weights}")
        print(f"   Threshold optimization: {self.optimize_thresholds}")

        # Use default features if none provided
        if feature_cols is None:
            feature_cols = [col for col in MODEL_FEATURES if col in df.columns]

        self.feature_cols = feature_cols
        print(f"   Using {len(feature_cols)} features")

        # Calculate base transition matrix with smoothing
        self._calculate_transition_matrix(df)

        # Train feature-dependent models with class weights
        if self.use_features and len(feature_cols) > 0:
            self._train_transition_models(df, feature_cols)
            self._train_wallet_share_models(df, feature_cols)

        # Optimize thresholds on validation data
        if self.optimize_thresholds and validation_data is not None:
            self._optimize_thresholds(validation_data)

        print(f"✅ Model training complete")

    def _calculate_transition_matrix(self, df: pd.DataFrame):
        """Calculate transition matrix with Laplace smoothing."""
        print(f"   📊 Calculating transition matrix...")

        # Count transitions
        transition_counts = pd.crosstab(df['state'], df['next_state'],
                                      margins=False, normalize=False)

        # Ensure all states are present
        for state in self.states:
            if state not in transition_counts.index:
                transition_counts.loc[state] = 0
            if state not in transition_counts.columns:
                transition_counts[state] = 0

        # Reorder to match self.states
        transition_counts = transition_counts.reindex(index=self.states,
                                                    columns=self.states,
                                                    fill_value=0)

        # Apply Laplace smoothing
        transition_counts_smooth = transition_counts + self.smoothing_alpha

        # Normalize to get probabilities
        self.transition_matrix = (transition_counts_smooth.div(
            transition_counts_smooth.sum(axis=1), axis=0)).values
        self.transition_counts = transition_counts.values

        print(f"   ✅ Transition matrix calculated")

    def _train_transition_models(self, df: pd.DataFrame, feature_cols: List[str]):
        """Train feature-dependent transition models with class weights."""
        print(f"   🎯 Training transition models with class weights...")

        # Prepare features
        X = df[feature_cols].fillna(0)

        # Train models for each current state
        for from_state in self.states:
            state_mask = df['state'] == from_state
            if state_mask.sum() < 10:  # Skip if too few samples
                continue

            X_state = X[state_mask]
            y_state = df[state_mask]['next_state']

            # Create class weight dict for sklearn
            sklearn_weights = {}
            for state in self.states:
                if state in y_state.values:
                    sklearn_weights[state] = self.class_weights.get(state, 1.0)

            # Train logistic regression with class weights
            model = LogisticRegression(
                class_weight=sklearn_weights,
                random_state=42,
                max_iter=1000
            )

            try:
                model.fit(X_state, y_state)
                self.transition_models[from_state] = model
                print(f"     ✅ Model trained for {from_state} → * transitions")
            except Exception as e:
                print(f"     ⚠️ Failed to train model for {from_state}: {e}")

    def _train_wallet_share_models(self, df: pd.DataFrame, feature_cols: List[str]):
        """Train wallet share prediction models."""
        print(f"   💰 Training wallet share models...")

        X = df[feature_cols].fillna(0)
        y = df['wallet_share_next']

        # Train models for each state
        for state in self.states:
            state_mask = df['state'] == state
            if state_mask.sum() < 10:
                continue

            X_state = X[state_mask]
            y_state = y[state_mask]

            # Use RandomForest for wallet share prediction
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Convert continuous to bins for classification
            y_binned = pd.cut(y_state, bins=[0, 0.2, 0.8, 1.0],
                             labels=['LOW', 'MED', 'HIGH'])

            try:
                model.fit(X_state, y_binned)
                self.wallet_share_models[state] = model
                print(f"     ✅ Wallet model trained for {state}")
            except Exception as e:
                print(f"     ⚠️ Failed to train wallet model for {state}: {e}")

    def _optimize_thresholds(self, validation_data: pd.DataFrame):
        """Optimize classification thresholds for better F1-LEAVE."""
        print(f"   🎯 Optimizing classification thresholds...")

        # Get baseline predictions
        baseline_probs = self.predict_proba(validation_data)
        y_true = validation_data['next_state']

        # Optimize threshold for LEAVE class
        leave_idx = list(self.states).index('LEAVE')
        leave_probs = baseline_probs[:, leave_idx]

        # Convert to binary classification for LEAVE
        y_true_binary = (y_true == 'LEAVE').astype(int)

        # Find optimal threshold using precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true_binary, leave_probs)

        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero

        # Find optimal threshold
        best_idx = np.argmax(f1_scores)
        optimal_leave_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5

        # Store optimized thresholds
        self.optimal_thresholds = {
            'STAY': 0.5,    # Keep default
            'SPLIT': 0.5,   # Keep default
            'LEAVE': max(0.2, min(0.8, optimal_leave_threshold))  # Bounded optimization
        }

        print(f"     ✅ Optimal LEAVE threshold: {self.optimal_thresholds['LEAVE']:.3f}")

        # Validate improvement
        self._validate_threshold_improvement(validation_data, y_true, baseline_probs)

    def _validate_threshold_improvement(self, validation_data, y_true, baseline_probs):
        """Validate that threshold optimization improves F1-LEAVE."""
        print(f"   📊 Validating threshold optimization...")

        # Baseline predictions (0.5 threshold)
        baseline_pred = np.argmax(baseline_probs, axis=1)
        baseline_pred_labels = [self.states[i] for i in baseline_pred]

        # Optimized predictions
        optimized_pred = self._apply_optimized_thresholds(baseline_probs)

        # Calculate metrics
        baseline_f1_leave = f1_score(y_true, baseline_pred_labels,
                                   labels=['LEAVE'], average='macro')
        optimized_f1_leave = f1_score(y_true, optimized_pred,
                                    labels=['LEAVE'], average='macro')

        baseline_accuracy = accuracy_score(y_true, baseline_pred_labels)
        optimized_accuracy = accuracy_score(y_true, optimized_pred)

        # Store validation metrics
        self.validation_metrics = {
            'baseline_f1_leave': baseline_f1_leave,
            'optimized_f1_leave': optimized_f1_leave,
            'f1_leave_improvement': optimized_f1_leave - baseline_f1_leave,
            'baseline_accuracy': baseline_accuracy,
            'optimized_accuracy': optimized_accuracy,
            'accuracy_change': optimized_accuracy - baseline_accuracy
        }

        print(f"     📈 F1-LEAVE improvement: {baseline_f1_leave:.3f} → {optimized_f1_leave:.3f} ({optimized_f1_leave - baseline_f1_leave:+.3f})")
        print(f"     📈 Accuracy change: {baseline_accuracy:.3f} → {optimized_accuracy:.3f} ({optimized_accuracy - baseline_accuracy:+.3f})")

    def _apply_optimized_thresholds(self, probabilities):
        """Apply optimized thresholds for classification."""
        predictions = []

        for prob_row in probabilities:
            # Check each class with its optimal threshold
            class_scores = {}
            for i, state in enumerate(self.states):
                threshold = self.optimal_thresholds.get(state, 0.5)
                class_scores[state] = prob_row[i] / threshold  # Normalized by threshold

            # Predict class with highest normalized score
            predicted_state = max(class_scores, key=class_scores.get)
            predictions.append(predicted_state)

        return predictions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict transition probabilities."""
        n_samples = len(df)
        probabilities = np.zeros((n_samples, len(self.states)))

        # Get base probabilities from transition matrix
        for i, current_state in enumerate(df['state']):
            if current_state in self.states:
                state_idx = self.states.index(current_state)
                probabilities[i] = self.transition_matrix[state_idx]

        # Adjust with feature-dependent models if available
        if self.use_features and self.transition_models:
            X = df[self.feature_cols].fillna(0)

            for i, current_state in enumerate(df['state']):
                if current_state in self.transition_models:
                    model = self.transition_models[current_state]
                    try:
                        feature_prob = model.predict_proba(X.iloc[[i]])
                        # Blend with base probabilities (80% feature, 20% base)
                        probabilities[i] = 0.8 * feature_prob[0] + 0.2 * probabilities[i]
                    except:
                        pass  # Keep base probabilities if error

        return probabilities

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with optimized thresholds."""
        probabilities = self.predict_proba(df)

        # Apply optimized thresholds if available
        if self.optimal_thresholds is not None:
            predicted_states = self._apply_optimized_thresholds(probabilities)
        else:
            # Default: argmax
            predicted_indices = np.argmax(probabilities, axis=1)
            predicted_states = [self.states[i] for i in predicted_indices]

        # Predict wallet share (simplified for now)
        wallet_forecasts = []
        for i, (_, row) in enumerate(df.iterrows()):
            current_wallet = row.get('wallet_share', 0.5)
            predicted_state = predicted_states[i]

            # Simple wallet share prediction based on state
            if predicted_state == 'STAY':
                forecast = min(1.0, current_wallet * 1.02)  # Slight increase
            elif predicted_state == 'SPLIT':
                forecast = current_wallet * 0.95  # Slight decrease
            else:  # LEAVE
                forecast = max(0.0, current_wallet * 0.3)  # Significant decrease

            wallet_forecasts.append(forecast)

        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': df['customer_id'],
            'next_state': predicted_states,
            'wallet_share_forecast': wallet_forecasts
        })

        return results

    def get_performance_summary(self):
        """Get summary of model improvements."""
        if not self.validation_metrics:
            return "No validation metrics available"

        summary = f"""
🎯 IMPROVED MODEL PERFORMANCE SUMMARY
=====================================

Class Weights Applied:
{self.class_weights}

Threshold Optimization:
• LEAVE threshold: {self.optimal_thresholds.get('LEAVE', 0.5):.3f}

Validation Results:
• F1-LEAVE: {self.validation_metrics['baseline_f1_leave']:.3f} → {self.validation_metrics['optimized_f1_leave']:.3f} ({self.validation_metrics['f1_leave_improvement']:+.3f})
• Accuracy: {self.validation_metrics['baseline_accuracy']:.3f} → {self.validation_metrics['optimized_accuracy']:.3f} ({self.validation_metrics['accuracy_change']:+.3f})

Trade-off Analysis:
• F1-LEAVE improvement: {self.validation_metrics['f1_leave_improvement']*100:+.1f}%
• Accuracy change: {self.validation_metrics['accuracy_change']*100:+.1f}%
• Net benefit: {'✅ Positive' if self.validation_metrics['f1_leave_improvement'] > abs(self.validation_metrics['accuracy_change'])/2 else '⚠️ Marginal'}
"""
        return summary
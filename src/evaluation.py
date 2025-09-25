"""Evaluation metrics and validation for KSCU Markov model."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional


def evaluate_state_predictions(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               labels: List[str] = ['STAY', 'SPLIT', 'LEAVE']) -> Dict:
    """
    Evaluate state prediction performance.
    
    Args:
        y_true: True state labels
        y_pred: Predicted state labels
        y_prob: Prediction probabilities (optional)
        labels: List of state labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    
    for i, label in enumerate(labels):
        metrics[f'precision_{label}'] = precision[i]
        metrics[f'recall_{label}'] = recall[i]
        metrics[f'f1_{label}'] = f1[i]
        metrics[f'support_{label}'] = support[i]
    
    # Weighted averages
    metrics['precision_weighted'] = np.average(precision, weights=support)
    metrics['recall_weighted'] = np.average(recall, weights=support)
    metrics['f1_weighted'] = np.average(f1, weights=support)
    
    # Log loss (if probabilities provided)
    if y_prob is not None:
        metrics['log_loss'] = log_loss(y_true, y_prob, labels=labels)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)
    
    return metrics


def evaluate_wallet_share_predictions(y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict:
    """
    Evaluate wallet share prediction performance.
    
    Args:
        y_true: True wallet share values
        y_pred: Predicted wallet share values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Ensure values are in [0, 1] range
    y_true = np.clip(y_true, 0, 1)
    y_pred = np.clip(y_pred, 0, 1)
    
    # Basic regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Percentage error metrics
    mask = y_true > 0.01  # Avoid division by zero
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['mape'] = mape
    
    # Correlation
    metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Directional accuracy (did we predict increase/decrease correctly?)
    if len(y_true) > 1:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(true_diff) == np.sign(pred_diff))
        metrics['directional_accuracy'] = directional_accuracy
    
    # Error by wallet share range
    ranges = [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            metrics[f'mae_{low}_{high}'] = mean_absolute_error(
                y_true[mask], y_pred[mask]
            )
            metrics[f'count_{low}_{high}'] = mask.sum()
    
    return metrics


def calculate_calibration_error(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate calibration error for probability predictions.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (calibration_error, fraction_positives, mean_predicted_prob)
    """
    fraction_positives, mean_predicted_prob = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    # Calculate expected calibration error (ECE)
    bin_counts, _ = np.histogram(y_prob, bins=n_bins)
    bin_weights = bin_counts / len(y_prob)
    
    # Only use bins with samples
    valid_bins = bin_counts > 0
    if valid_bins.sum() > 0:
        calibration_error = np.average(
            np.abs(fraction_positives - mean_predicted_prob),
            weights=bin_weights[valid_bins]
        )
    else:
        calibration_error = 0.0
    
    return calibration_error, fraction_positives, mean_predicted_prob


def evaluate_transition_matrix(true_transitions: pd.DataFrame,
                              pred_transitions: pd.DataFrame,
                              states: List[str] = ['STAY', 'SPLIT', 'LEAVE']) -> Dict:
    """
    Evaluate transition matrix predictions.
    
    Args:
        true_transitions: True transition counts/probabilities
        pred_transitions: Predicted transition counts/probabilities
        states: List of states
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Normalize to probabilities
    true_probs = true_transitions.div(true_transitions.sum(axis=1), axis=0)
    pred_probs = pred_transitions.div(pred_transitions.sum(axis=1), axis=0)
    
    # Frobenius norm of difference
    diff = true_probs.values - pred_probs.values
    metrics['frobenius_norm'] = np.linalg.norm(diff, 'fro')
    
    # KL divergence for each row
    kl_divs = []
    for state in states:
        if state in true_probs.index and state in pred_probs.index:
            true_row = true_probs.loc[state].values + 1e-10
            pred_row = pred_probs.loc[state].values + 1e-10
            kl = np.sum(true_row * np.log(true_row / pred_row))
            kl_divs.append(kl)
            metrics[f'kl_divergence_{state}'] = kl
    
    metrics['mean_kl_divergence'] = np.mean(kl_divs) if kl_divs else 0
    
    return metrics


def plot_evaluation_results(metrics: Dict, save_path: Optional[str] = None):
    """
    Create visualization of evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confusion Matrix
    if 'confusion_matrix' in metrics:
        ax = axes[0, 0]
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('State Prediction Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # 2. Per-class F1 scores
    ax = axes[0, 1]
    states = ['STAY', 'SPLIT', 'LEAVE']
    f1_scores = [metrics.get(f'f1_{s}', 0) for s in states]
    ax.bar(states, f1_scores, color=['green', 'orange', 'red'])
    ax.set_title('F1 Score by State')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 3. Wallet Share MAE by range
    ax = axes[0, 2]
    ranges = ['0-0.2', '0.2-0.5', '0.5-0.8', '0.8-1.0']
    mae_values = []
    for r in ranges:
        key = f"mae_{r.replace('-', '_').replace('.', '_')}"
        mae_values.append(metrics.get(key, 0))
    
    if any(mae_values):
        ax.bar(ranges, mae_values, color='steelblue')
        ax.set_title('Wallet Share MAE by Range')
        ax.set_xlabel('Wallet Share Range')
        ax.set_ylabel('MAE')
        for i, v in enumerate(mae_values):
            if v > 0:
                ax.text(i, v + 0.001, f'{v:.3f}', ha='center')
    
    # 4. Overall metrics summary
    ax = axes[1, 0]
    summary_text = f"""
    State Prediction:
    - Accuracy: {metrics.get('accuracy', 0):.3f}
    - Log Loss: {metrics.get('log_loss', 0):.3f}
    - F1 (weighted): {metrics.get('f1_weighted', 0):.3f}
    
    Wallet Share:
    - MAE: {metrics.get('mae', 0):.4f}
    - RMSE: {metrics.get('rmse', 0):.4f}
    - Correlation: {metrics.get('correlation', 0):.3f}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
    ax.set_title('Overall Performance Summary')
    ax.axis('off')
    
    # 5. Calibration plot (if available)
    ax = axes[1, 1]
    if 'calibration_curve' in metrics:
        frac_pos, mean_pred = metrics['calibration_curve']
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', label='Model')
        ax.set_title('Calibration Plot')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Feature importance (placeholder)
    ax = axes[1, 2]
    if 'feature_importance' in metrics:
        importance = metrics['feature_importance']
        top_features = importance.head(10)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_title('Top 10 Feature Importances')
        ax.set_xlabel('Importance')
    
    plt.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Evaluation plot saved to {save_path}")
    
    plt.show()


def cross_validate_markov(model_class, df: pd.DataFrame, 
                         n_folds: int = 5, 
                         feature_cols: Optional[List[str]] = None) -> Dict:
    """
    Perform cross-validation for Markov model.
    
    Args:
        model_class: Markov model class
        df: DataFrame with data
        n_folds: Number of CV folds
        feature_cols: Feature columns to use
        
    Returns:
        Dictionary of CV results
    """
    from sklearn.model_selection import KFold
    
    # Initialize results storage
    cv_results = {
        'state_accuracy': [],
        'state_log_loss': [],
        'wallet_mae': [],
        'wallet_rmse': []
    }
    
    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    customer_ids = df['customer_id'].unique()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(customer_ids)):
        print(f"Fold {fold + 1}/{n_folds}")
        
        # Split by customer ID to avoid leakage
        train_customers = customer_ids[train_idx]
        val_customers = customer_ids[val_idx]
        
        train_data = df[df['customer_id'].isin(train_customers)]
        val_data = df[df['customer_id'].isin(val_customers)]
        
        # Train model
        model = model_class()
        model.fit(train_data, feature_cols)
        
        # Evaluate on validation set
        val_predictions = []
        val_probs = []
        wallet_predictions = []
        
        for _, row in val_data.iterrows():
            features = row[feature_cols].to_frame().T if feature_cols else None
            
            # Predict state transition
            trans_probs = model.predict_transition_probs(row['state'], features)
            pred_state = max(trans_probs, key=trans_probs.get)
            val_predictions.append(pred_state)
            
            # Get probabilities for log loss
            prob_vector = [trans_probs.get(s, 0) for s in model.states]
            val_probs.append(prob_vector)
            
            # Predict wallet share
            wallet_pred = model.predict_wallet_share(row['next_state'], features)
            wallet_predictions.append(wallet_pred)
        
        # Calculate metrics
        state_metrics = evaluate_state_predictions(
            val_data['next_state'].values,
            np.array(val_predictions),
            np.array(val_probs) if val_probs else None,
            labels=model.states
        )
        
        wallet_metrics = evaluate_wallet_share_predictions(
            val_data['wallet_share_next'].values,
            np.array(wallet_predictions)
        )
        
        # Store results
        cv_results['state_accuracy'].append(state_metrics['accuracy'])
        if 'log_loss' in state_metrics:
            cv_results['state_log_loss'].append(state_metrics['log_loss'])
        cv_results['wallet_mae'].append(wallet_metrics['mae'])
        cv_results['wallet_rmse'].append(wallet_metrics['rmse'])
    
    # Summarize results
    summary = {}
    for metric, values in cv_results.items():
        if values:
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("- evaluate_state_predictions()")
    print("- evaluate_wallet_share_predictions()")
    print("- calculate_calibration_error()")
    print("- evaluate_transition_matrix()")
    print("- plot_evaluation_results()")
    print("- cross_validate_markov()")
"""Model tuning and optimization for better performance."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
import warnings
warnings.filterwarnings('ignore')

try:
    from .markov_model import MarkovChainModel
    from .evaluation import evaluate_state_predictions, evaluate_wallet_share_predictions
    from .config import MODEL_FEATURES
except ImportError:
    from markov_model import MarkovChainModel
    from evaluation import evaluate_state_predictions, evaluate_wallet_share_predictions
    from config import MODEL_FEATURES


class ImprovedMarkovModel(MarkovChainModel):
    """Enhanced Markov model with better probability calibration."""
    
    def __init__(self, 
                 smoothing_alpha: float = 0.01,
                 use_features: bool = True,
                 class_weight: str = 'balanced'):
        super().__init__(smoothing_alpha, use_features)
        self.class_weight = class_weight
        
    def _train_transition_models(self, df: pd.DataFrame, feature_cols: list):
        """Train feature-dependent transition models with better calibration."""
        
        for from_state in self.states:
            state_data = df[df['state'] == from_state].copy()
            
            if len(state_data) < 50:
                continue
            
            # Prepare features and target
            X = state_data[feature_cols].fillna(0)
            y = state_data['next_state']
            
            # Use Logistic Regression for better probability calibration
            # and handle class imbalance
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight=self.class_weight,
                solver='liblinear'  # Better for small datasets
            )
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            self.transition_models[from_state] = {
                'model': model,
                'scaler': scaler,
                'features': feature_cols,
                'classes': model.classes_
            }
            
            print(f"Trained transition model for {from_state} -> ? (accuracy: {model.score(X_scaled, y):.3f})")
    
    def predict_transition_probs(self, current_state: str, features: pd.DataFrame = None) -> dict:
        """Predict transition probabilities with improved calibration."""
        
        if not self.use_features or features is None or current_state not in self.transition_models:
            # Use base transition matrix
            if current_state in self.transition_matrix.index:
                probs = self.transition_matrix.loc[current_state].to_dict()
            else:
                probs = {state: 1.0 / len(self.states) for state in self.states}
        else:
            model_info = self.transition_models[current_state]
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Prepare features
            X = features[model_info['features']].fillna(0)
            X_scaled = scaler.transform(X.values.reshape(1, -1))
            
            # Get probabilities
            prob_array = model.predict_proba(X_scaled)[0]
            probs = dict(zip(model_info['classes'], prob_array))
            
            # Ensure all states have probabilities
            for state in self.states:
                if state not in probs:
                    probs[state] = 0.001  # Small non-zero probability
            
            # Apply smoothing to avoid extreme probabilities
            total = sum(probs.values())
            smoothing = 0.01
            probs = {k: (v + smoothing) / (total + len(probs) * smoothing) 
                    for k, v in probs.items()}
        
        return probs


def tune_hyperparameters(train_data: pd.DataFrame, 
                        val_data: pd.DataFrame,
                        feature_cols: list) -> dict:
    """
    Tune hyperparameters for the Markov model.
    
    Returns:
        Best parameters and performance metrics
    """
    
    print("Tuning hyperparameters...")
    
    # Parameter grid
    param_grid = {
        'smoothing_alpha': [0.001, 0.01, 0.1],
        'class_weight': ['balanced', None],
    }
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for smoothing in param_grid['smoothing_alpha']:
        for class_weight in param_grid['class_weight']:
            
            print(f"Testing: smoothing={smoothing}, class_weight={class_weight}")
            
            # Train model
            model = ImprovedMarkovModel(
                smoothing_alpha=smoothing,
                class_weight=class_weight
            )
            model.fit(train_data, feature_cols)
            
            # Evaluate on validation
            val_predictions = []
            val_probs = []
            
            for _, row in val_data.iterrows():
                features = row[feature_cols].to_frame().T
                trans_probs = model.predict_transition_probs(row['state'], features)
                
                pred_state = max(trans_probs, key=trans_probs.get)
                val_predictions.append(pred_state)
                
                prob_vector = [trans_probs.get(s, 0) for s in model.states]
                val_probs.append(prob_vector)
            
            # Calculate metrics
            try:
                current_log_loss = log_loss(val_data['next_state'], val_probs, labels=model.states)
                
                state_metrics = evaluate_state_predictions(
                    val_data['next_state'].values,
                    np.array(val_predictions),
                    np.array(val_probs),
                    labels=model.states
                )
                
                # Combined score (emphasize log loss and LEAVE F1)
                leave_f1 = state_metrics.get('f1_LEAVE', 0)
                score = current_log_loss + (1 - leave_f1) * 2  # Penalty for poor LEAVE F1
                
                results.append({
                    'smoothing_alpha': smoothing,
                    'class_weight': class_weight,
                    'log_loss': current_log_loss,
                    'leave_f1': leave_f1,
                    'accuracy': state_metrics['accuracy'],
                    'combined_score': score
                })
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'smoothing_alpha': smoothing,
                        'class_weight': class_weight
                    }
                
                print(f"  LogLoss: {current_log_loss:.3f}, LEAVE F1: {leave_f1:.3f}, Score: {score:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    print(f"\\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.3f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


def train_ensemble_model(train_data: pd.DataFrame, feature_cols: list):
    """Train an ensemble of models for better performance."""
    
    print("Training ensemble model...")
    
    models = {}
    
    # For each state transition, train multiple models
    for from_state in ['STAY', 'SPLIT', 'LEAVE']:
        state_data = train_data[train_data['state'] == from_state].copy()
        
        if len(state_data) < 50:
            continue
        
        X = state_data[feature_cols].fillna(0)
        y = state_data['next_state']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train multiple models
        model_ensemble = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        lr.fit(X_scaled, y)
        model_ensemble['logistic'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_scaled, y)
        model_ensemble['forest'] = rf
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_scaled, y)
        model_ensemble['gradient'] = gb
        
        models[from_state] = {
            'ensemble': model_ensemble,
            'scaler': scaler,
            'features': feature_cols
        }
        
        print(f"Trained ensemble for {from_state} -> ?")
    
    return models


def evaluate_with_improved_metrics(model, test_data: pd.DataFrame, feature_cols: list):
    """Comprehensive evaluation with focus on competition metrics."""
    
    print("Running comprehensive evaluation...")
    
    # Predict on test set
    test_predictions = []
    test_probs = []
    wallet_predictions = []
    
    for _, row in test_data.iterrows():
        features = row[feature_cols].to_frame().T
        
        # State prediction
        trans_probs = model.predict_transition_probs(row['state'], features)
        pred_state = max(trans_probs, key=trans_probs.get)
        test_predictions.append(pred_state)
        
        # Probabilities
        prob_vector = [trans_probs.get(s, 0) for s in model.states]
        test_probs.append(prob_vector)
        
        # Wallet share
        wallet_pred = model.predict_wallet_share(row['next_state'], features)
        wallet_predictions.append(wallet_pred)
    
    # Evaluate predictions
    state_metrics = evaluate_state_predictions(
        test_data['next_state'].values,
        np.array(test_predictions),
        np.array(test_probs),
        labels=model.states
    )
    
    wallet_metrics = evaluate_wallet_share_predictions(
        test_data['wallet_share_next'].values,
        np.array(wallet_predictions)
    )
    
    # Competition-specific analysis
    print("\\n" + "="*50)
    print("COMPETITION PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Primary metrics (60% of score)
    print("\\n1. PREDICTIVE QUALITY (60% weight):")
    print(f"   • LogLoss: {state_metrics['log_loss']:.3f} (Target: < 0.5)")
    print(f"   • Wallet MAE: {wallet_metrics['mae']:.4f} (Target: < 0.15)")
    print(f"   • Calibration needed: {'Yes' if state_metrics['log_loss'] > 0.5 else 'No'}")
    
    # Per-class performance
    print("\\n2. STATE-SPECIFIC PERFORMANCE:")
    for state in model.states:
        f1 = state_metrics.get(f'f1_{state}', 0)
        precision = state_metrics.get(f'precision_{state}', 0)
        recall = state_metrics.get(f'recall_{state}', 0)
        print(f"   • {state:5}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Business value metrics (25% of score)
    print("\\n3. BUSINESS VALUE INDICATORS:")
    leave_detection = state_metrics.get('recall_LEAVE', 0)
    print(f"   • LEAVE detection rate: {leave_detection:.3f} (critical for retention)")
    print(f"   • False positive rate: {1 - state_metrics.get('precision_STAY', 0):.3f}")
    print(f"   • Overall accuracy: {state_metrics['accuracy']:.3f}")
    
    return {
        'state_metrics': state_metrics,
        'wallet_metrics': wallet_metrics,
        'competition_score': _calculate_competition_score(state_metrics, wallet_metrics)
    }


def _calculate_competition_score(state_metrics: dict, wallet_metrics: dict) -> dict:
    """Calculate estimated competition score."""
    
    # Predictive quality (60%)
    log_loss_score = max(0, 1 - state_metrics['log_loss'] / 2)  # Normalize
    wallet_score = max(0, 1 - wallet_metrics['mae'] / 0.15)  # Normalize by target
    predictive_score = (log_loss_score + wallet_score) / 2
    
    # Business value (25%) - based on leave detection and overall accuracy
    leave_f1 = state_metrics.get('f1_LEAVE', 0)
    business_score = (leave_f1 + state_metrics['accuracy']) / 2
    
    # Application delivery (15%) - assume moderate score for now
    delivery_score = 0.7
    
    # Weighted total
    total_score = (predictive_score * 0.6 + 
                   business_score * 0.25 + 
                   delivery_score * 0.15)
    
    return {
        'predictive_quality': predictive_score,
        'business_value': business_score,
        'delivery': delivery_score,
        'total_estimated': total_score
    }


if __name__ == "__main__":
    # Load data
    train = pd.read_csv('../data/splits/train.csv')
    val = pd.read_csv('../data/splits/val.csv')
    test = pd.read_csv('../data/splits/test.csv')
    
    feature_cols = [col for col in MODEL_FEATURES if col in train.columns]
    
    # Tune hyperparameters
    tuning_results = tune_hyperparameters(train, val, feature_cols)
    
    # Train best model
    best_params = tuning_results['best_params']
    final_model = ImprovedMarkovModel(**best_params)
    final_model.fit(train, feature_cols)
    
    # Evaluate on test set
    test_results = evaluate_with_improved_metrics(final_model, test, feature_cols)
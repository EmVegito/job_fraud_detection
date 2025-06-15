import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Import your preprocessor
from src.data_processing.preprocessor import JobFraudDataPipeline


class ModelTrainingPipeline:
    """
    Comprehensive model training pipeline that handles multiple algorithms,
    hyperparameter tuning, class imbalance, and model evaluation.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 cv_folds: int = 5,
                 scoring_metric: str = 'f1',
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize the model training pipeline.
        
        Args:
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
            scoring_metric: Primary metric for model selection ('f1', 'precision', 'recall', 'roc_auc')
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to print progress information
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize containers
        self.models = {}
        self.param_grids = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.evaluation_results = {}
        
        # Setup default models and parameter grids
        self._setup_models()
        
    def _setup_models(self):
        """Setup default models and their hyperparameter grids with proper regularization."""
        
        # Define models with regularization to prevent overfitting
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced',
                fit_intercept=False,
                solver='liblinear'
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced',
            ),
            'ada_boost': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=42, criterion='log_loss', class_weight='balanced'),
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                bootstrap=True, 
                oob_score=True,  
                max_samples=0.8,
                class_weight="balanced",
                max_features=None,
                max_depth=None,
                criterion='log_loss',
                min_samples_leaf=1,
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                learning_rate=0.1,
                max_depth=10,
                validation_fraction=0.1,  
                n_iter_no_change=10,  
                tol=1e-4, 
                subsample=0.8,
                max_features=None,
                loss='log_loss'
            ),
        }
        
        # Define parameter grids with focus on regularization
        self.param_grids = {
            'logistic_regression': {
                'penalty': ['l1', 'l2'], 
                'C': [0.1, 1, 100],    
            },
            'svm': {
                'C': [0.1, 1.0, 10],  
                'kernel': ['rbf', 'linear'],
                'gamma': [0.01, 0.1],
            },
            'ada_boost': {
                'n_estimators': [200, 300], 
                'learning_rate': [0.05, 0.1], 
                'estimator__max_depth': [5, None],
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'min_samples_split': [2, 3],
            },
            'gradient_boosting': {
                'n_estimators': [300, 400], 
                'min_samples_split': [5, 10],
                'min_samples_leaf': [4, 6],      
            }
        }
    
    def add_custom_model(self, name: str, model, param_grid: Dict[str, Any]):
        """
        Add a custom model to the pipeline.
        
        Args:
            name: Name of the model
            model: Sklearn-compatible model instance
            param_grid: Parameter grid for hyperparameter tuning
        """
        self.models[name] = model
        self.param_grids[name] = param_grid
        if self.verbose:
            print(f"Added custom model: {name}")
    
    def train_models(self, 
                    X_train, 
                    y_train,
                    search_type: str = 'grid',
                    models_to_train: Optional[list] = None) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: Strategy for handling class imbalance
            search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            models_to_train: List of model names to train (None for all)
            
        Returns:
            Dictionary with training results
        """
        if self.verbose:
            print("="*60)
            print("STARTING MODEL TRAINING PIPELINE")
            print("="*60)
        
        # Determine which models to train
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        training_results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                print(f"Warning: Model '{model_name}' not found. Skipping...")
                continue
                
            if self.verbose:
                print(f"\n{'='*40}")
                print(f"Training: {model_name.upper()}")
                print(f"{'='*40}")
            
            try:
                # Get model and parameter grid
                model = self.models[model_name]
                param_grid = self.param_grids[model_name]

                cv_fold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                
                # Choose search strategy
                if search_type == 'grid':
                    search = GridSearchCV(
                        model, 
                        param_grid,
                        cv=cv_fold,
                        scoring=self.scoring_metric,
                        n_jobs=self.n_jobs,
                        verbose=1 if self.verbose else 0
                    )
                else:  # random search
                    search = RandomizedSearchCV(
                        model,
                        param_grid,
                        n_iter=25,
                        cv=cv_fold,
                        scoring=self.scoring_metric,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        verbose=1 if self.verbose else 0
                    )

                # Fit the model
                search.fit(X_train, y_train)
                
                # Store results
                self.trained_models[model_name] = search.best_estimator_
                self.model_scores[model_name] = search.best_score_
                
                training_results[model_name] = {
                    'best_estimator': search.best_estimator_,
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'cv_results': search.cv_results_
                }
                
                if self.verbose:
                    print(f"Best {self.scoring_metric}: {search.best_score_:.4f}")
                    print(f"Best params: {search.best_params_}")
                    
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model
        if self.model_scores:
            self.best_model_name = max(self.model_scores, key=self.model_scores.get)
            self.best_model = self.trained_models[self.best_model_name]
            self.best_params = training_results[self.best_model_name]['best_params']
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"BEST MODEL: {self.best_model_name.upper()}")
                print(f"Best {self.scoring_metric}: {self.model_scores[self.best_model_name]:.4f}")
                print(f"{'='*60}")
        
        return training_results
    
    def evaluate_models(self, X_valid, y_valid) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on validation data.
        
        Args:
            X_valid: Validation features
            y_valid: Validation labels
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        if not self.trained_models:
            raise ValueError("No trained models found. Run train_models() first.")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("MODEL EVALUATION ON VALIDATION DATA")
            print(f"{'='*60}")
        
        evaluation_results = {}
        
        for model_name, model in self.trained_models.items():
            if self.verbose:
                print(f"\nEvaluating: {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_valid)
            y_pred_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_valid, y_pred),
                'precision': precision_score(y_valid, y_pred),
                'recall': recall_score(y_valid, y_pred),
                'f1_score': f1_score(y_valid, y_pred),  # For binary classification
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_valid, y_pred_proba)
            
            evaluation_results[model_name] = metrics
            
            if self.verbose:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def get_detailed_report(self, X_valid, y_valid, model_name: str = None):
        """
        Get detailed classification report and confusion matrix.
        
        Args:
            X_valid: Validation features
            y_valid: Validation labels
            model_name: Specific model name (None for best model)
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        y_pred = model.predict(X_valid)
        
        print(f"\nDETAILED REPORT FOR: {model_name.upper()}")
        print("="*50)
        print("\nClassification Report:")
        print(classification_report(y_valid, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_valid, y_pred))
    
    def save_best_model(self, filepath: str):
        """
        Save the best performing model to a pickle file.
        
        Args:
            filepath: Path where to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run train_models() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model along with metadata
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'best_params': self.best_params,
            'best_score': self.model_scores[self.best_model_name],
            'scoring_metric': self.scoring_metric,
            'all_scores': self.model_scores,
            'evaluation_results': self.evaluation_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            print(f"\nBest model saved to: {filepath}")
            print(f"Model: {self.best_model_name}")
            print(f"Best {self.scoring_metric}: {self.model_scores[self.best_model_name]:.4f}")
    
    @staticmethod
    def load_model(filepath: str):
        """
        Load a saved model from pickle file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Dictionary containing model and metadata
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison dataframe of all model performances.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results found. Run evaluate_models() first.")
        
        comparison_df = pd.DataFrame(self.evaluation_results).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        return comparison_df


# Example usage function
def example_usage():
    """
    Example of how to use the ModelTrainingPipeline class.
    """
    # Load your dataset
    df = pd.read_csv('./data/raw/training_data.csv')
    
    # Initialize and run preprocessing pipeline
    preprocessor = JobFraudDataPipeline(
        max_tfidf_features=500,
        test_size=0.2,
        random_state=42
    )
    
    # Transform the data
    X_train, X_valid, y_train, y_valid = preprocessor.fit_transform(df)
    
    # Save preprocessor
    preprocessor.save_pipeline('./data/models/preprocessor.pkl')
    
    # Initialize model training pipeline
    model_trainer = ModelTrainingPipeline(
        random_state=42,
        cv_folds=5,
        scoring_metric='f1',
        verbose=True
    )
    
    # Train models
    training_results = model_trainer.train_models(
        X_train, y_train,
        sampling_strategy='smote',
        search_type='grid',
        models_to_train=['logistic_regression', 'random_forest', 'gradient_boosting']
    )
    
    # Evaluate models
    evaluation_results = model_trainer.evaluate_models(X_valid, y_valid)
    
    # Get detailed report for best model
    model_trainer.get_detailed_report(X_valid, y_valid)
    
    # Get model comparison
    comparison_df = model_trainer.get_model_comparison()
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save best model
    model_trainer.save_best_model('./data/models/best_model.pkl')
    
    return model_trainer, comparison_df

if __name__ == "__main__":
    # Run example
    trainer, comparison = example_usage()
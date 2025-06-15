import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing.preprocessor import JobFraudDataPipeline
from src.models.model_trainer import ModelTrainingPipeline  # Your new class
import os
import gdown
import sys
import pickle

from src.data_processing.data_loader import download_google_drive_csv
from src.logger import logging
from src.exception import CustomException
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main function that runs the complete ML pipeline:
    1. Data preprocessing
    2. Model training with hyperparameter tuning
    3. Model evaluation
    4. Save best model
    """
    print("="*80)
    print("JOB FRAUD DETECTION - COMPLETE ML PIPELINE")
    print("="*80)
    
    # ==========================================
    # 1. DATA DOWNLOADING LOCALLY
    # ==========================================

    google_drive_file_id = ["1nxyKfzFkbasXYyNIQPqbdZR69ygJ6orT", "15Y-u50AAvuJb1hIOPJzU8ccsdwTDEBJR"]
    output_filename = ["testing_data.csv", "training_data.csv"]
    output_directory = "./data/raw"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_id, filename in zip(google_drive_file_id, output_filename):
        full_output_path = os.path.join(output_directory, filename)

        try:
            logging.info(f"Downloading {filename}")
            download_google_drive_csv(file_id, full_output_path)

            import pandas as pd
            df = pd.read_csv(full_output_path)
            print(f"\n{filename} Shape:")
            print(df.shape)
        
        except FileNotFoundError:
            print(f"\nCould not find the downloaded file at {full_output_path}. Check for download errors.")
        except Exception as e:
            raise CustomException(e, sys)

    print("\n🔄 STEP 1: DATA PREPROCESSING")
    print("-" * 40)

    # ==========================================
    # 2. DATA PREPROCESSING
    # ==========================================
    
    # Load dataset

    logging.info("Loadin Training CSV as DataFrame.")
    df = pd.read_csv('./data/raw/training_data.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize preprocessing pipeline
    # Initialize pipeline
    logging.info("Initializing Preprocessor for training")
    preprocessor = JobFraudDataPipeline(
        max_tfidf_features=180,
        outlier_method='iqr',
        outlier_threshold_multiplier=1.5
    )

    logging.info("Fitting Pipeline.")
    # This now properly prevents data leakage
    X_train, X_valid, y_train, y_valid = preprocessor.fit_transform(df)
    logging.info("Pipeline Fitted.")


    # Save preprocessor for later use
    preprocessor_path = './data/models/preprocessor.pkl'
    preprocessor.save_pipeline(preprocessor_path)
    logging.info("Preprocessor Saved")
    print(f"✅ Preprocessor saved to: {preprocessor_path}")
    
    # ==========================================
    # 3. MODEL TRAINING
    # ==========================================
    print("\n🤖 STEP 2: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("-" * 40)
    
    # Initialize model training pipeline
    logging.info("Initializing Model Trainer Object for training models.")
    model_trainer = ModelTrainingPipeline(
        random_state=42,
        cv_folds=3,
        scoring_metric='f1',
        n_jobs=-1,
        verbose=True
    )
    try:
        logging.info("Started Model Training")
        # Train multiple models with hyperparameter tuning
        training_results = model_trainer.train_models(
            X_train, 
            y_train,
            search_type='grid',   
            models_to_train=[       
                'logistic_regression',
                'random_forest',
                'ada_boost',
                'gradient_boosting',
                'svm',
            ]
        )
    except Exception as e:
        raise CustomException(e, sys)
    
    
    # ==========================================
    # 4. MODEL EVALUATION
    # ==========================================
    print("\n📊 STEP 3: MODEL EVALUATION")
    print("-" * 40)
    
    # Evaluate all models on validation set
    logging.info("Evaluation of Models Started")
    evaluation_results = model_trainer.evaluate_models(X_valid, y_valid)
    
    # Get model comparison table
    comparison_df = model_trainer.get_model_comparison()
    print("\n📈 MODEL PERFORMANCE COMPARISON:")
    print(comparison_df.to_string())
    
    # Get detailed report for best model
    print(f"\n🏆 DETAILED REPORT FOR BEST MODEL:")
    model_trainer.get_detailed_report(X_valid, y_valid)
    logging.info('All metric Evaluated for all models')
    # ==========================================
    # 5. SAVE BEST MODEL
    # ==========================================
    print("\n💾 STEP 4: SAVING BEST MODEL")
    print("-" * 40)
    try:
        logging.info("Saving best model")
        # Save the best performing model
        best_model_path = './data/models/best_model.pkl'
        model_trainer.save_best_model(best_model_path)
    except Exception as e:
        raise CustomException(e, sys)
    # ==========================================
    # 6. SUMMARY
    # ==========================================
    print("\n📋 PIPELINE SUMMARY")
    print("=" * 40)
    print(f"✅ Best Model: {model_trainer.best_model_name}")
    print(f"✅ Best F1-Score: {model_trainer.model_scores[model_trainer.best_model_name]:.4f}")
    print(f"✅ Models trained: {len(model_trainer.trained_models)}")
    print(f"✅ Preprocessor saved: {preprocessor_path}")
    print(f"✅ Best model saved: {best_model_path}")
    
    # Return objects for further analysis if needed
    return model_trainer, comparison_df, preprocessor


def predict_on_test_data(test_data_path: str = './data/raw/testing_data.csv'):
    """
    Function to make predictions on test data using saved models.
    
    Args:
        test_data_path: Path to test dataset
    
    Returns:
        Predictions and probabilities
    """
    print("\n🔮 MAKING PREDICTIONS ON TEST DATA")
    print("-" * 40)
    
    try:
        logging.info('Predicting on testing_data.csv')
        # Load test data
        df_test = pd.read_csv(test_data_path)
        print(f"Test dataset loaded: {df_test.shape}")
        
        # Load saved preprocessor
        preprocessor = JobFraudDataPipeline.load_pipeline('./data/models/preprocessor.pkl')
        print("✅ Preprocessor loaded")
        
        # Transform test data
        X_test = preprocessor.transform_test_data(df_test)
        print(f"Test data transformed: {X_test.shape}")
        
        # Load saved model
        model_data = ModelTrainingPipeline.load_model('./data/models/best_model.pkl')
        best_model = model_data['model']
        model_name = model_data['model_name']
        print(f"✅ Best model loaded: {model_name}")
        
        # Make predictions
        test_predictions = best_model.predict(X_test)
        test_probabilities = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        print(f"✅ Predictions completed")
        print(f"   - Fraud cases predicted: {sum(test_predictions)}")
        print(f"   - Non-fraud cases predicted: {len(test_predictions) - sum(test_predictions)}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'job_id': range(len(test_predictions)),
            'prediction': test_predictions,
            'fraud_probability': test_probabilities if test_probabilities is not None else np.nan
        })
        
        predictions_path = './data/predictions/test_predictions.csv'
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✅ Predictions saved to: {predictions_path}")
        
        return test_predictions, test_probabilities, predictions_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("  1. Test data at the specified path")
        print("  2. Saved preprocessor at './data/models/preprocessor.pkl'")
        print("  3. Saved model at './data/models/best_model.pkl'")
        return None, None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None, None


def analyze_model_performance(model_trainer, comparison_df):
    """
    Additional analysis of model performance.
    
    Args:
        model_trainer: Trained ModelTrainingPipeline instance
        comparison_df: Model comparison DataFrame
    """
    print("\n🔍 ADDITIONAL PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Top 3 models
    top_3_models = comparison_df.head(3)
    print("🏆 TOP 3 MODELS BY F1-SCORE:")
    print(top_3_models[['f1_score', 'accuracy', 'precision', 'recall']].to_string())
    
    # Model insights
    print(f"\n💡 INSIGHTS:")
    best_model_name = comparison_df.index[0]
    best_f1 = comparison_df.loc[best_model_name, 'f1_score']
    
    print(f"• Best performing model: {best_model_name}")
    print(f"• Best F1-score: {best_f1:.4f}")
    
    if best_f1 > 0.8:
        print("• 🎉 Excellent performance! Model is ready for deployment.")
    elif best_f1 > 0.7:
        print("• 👍 Good performance! Consider further optimization.")
    else:
        print("• ⚠️  Room for improvement. Consider:")
        print("    - Feature engineering")
        print("    - Different sampling strategies") 
        print("    - Advanced models (XGBoost, LightGBM)")


# ==========================================
# # . TRAIN MODEL ON WHOLE DATASET
# ==========================================

def train_model_on_full_data(preprocessor_path: str = './data/models/preprocessor.pkl', 
                             model_path:str = './data/models/best_model.pkl', 
                             train_data_path:str= './data/raw/training_data.csv'):
    """
    Trains the best model on full dataset

    Args:
        model_path: the path to the best_model.pkl
        train_data_path: the path to training_data.csv
    """
    try:
        print("\n🔮 TRAINING BEST MODEL ON FULL DATASET")
        print("-" * 40)
        
        #Load downloaded training data
        logging.info("Loading Training Data")
        df_train = pd.read_csv(train_data_path)
        print(f"Train dataset loaded: {df_train.shape}")
        
        # Load saved preprocessor
        logging.info("Loading Preprocessor")
        preprocessor = JobFraudDataPipeline.load_pipeline(preprocessor_path)
        print("✅ Preprocessor loaded")

        # Load saved best model
        logging.info("Loading Best model")
        model_data = ModelTrainingPipeline.load_model(model_path)

        logging.info("Preprocessing the whole training Data")
        X_train_scaled, df_train_processed = preprocessor._process_data(df_train, fit=True)
        y_train = df_train_processed['fraudulent']

        logging.info("Saving Preprocesssor")
        preprocessor.save_pipeline(preprocessor_path)
        print(f"✅ Preprocessor saved to: {preprocessor_path}")

        final_model = model_data['model']

        logging.info("Training the best model on full training data")
        final_model.fit(X_train_scaled, y_train)

        with open('./data/models/final_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        print(f"✅ Final Model saved to: ./data/models/final_model.pkl")
        logging.info('Final Fully trained Model Saved')
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Run the complete pipeline
    model_trainer, comparison_df, preprocessor = main()
    
    # Additional analysis
    analyze_model_performance(model_trainer, comparison_df)

    test_predictions, test_probabilities, predictions_df = predict_on_test_data()
    
    train_model_on_full_data()
    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
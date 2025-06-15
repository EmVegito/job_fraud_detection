import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

from src.logger import logging
from src.data_processing.preprocessor import JobFraudDataPipeline


class FraudJobDetector:
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize the fraud job detector
        
        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the trained preprocessor (JobFraudDataPipeline)
        """
        self.model = None
        self.preprocessor = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if preprocessor_path and os.path.exists(preprocessor_path):
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def load_preprocessor(self, preprocessor_path):
        """Load the trained preprocessor pipeline"""
        try:
            self.preprocessor = JobFraudDataPipeline.load_pipeline(preprocessor_path)
            logging.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        except Exception as e:
            logging.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def preprocess_job_posting(self, job_data):
        """
        Preprocess job posting data for prediction using the trained preprocessor pipeline
        
        Args:
            job_data: Dictionary containing job posting information
            
        Returns:
            Preprocessed features ready for model prediction
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        try:
            # Convert job_data dictionary to DataFrame format expected by preprocessor
            df = pd.DataFrame([job_data])
            
            # Ensure all expected columns are present with default values
            expected_columns = [
                'job_id', 'title', 'location', 'department', 'salary_range',
                'company_profile', 'description', 'requirements', 'benefits',
                'telecommuting', 'has_company_logo', 'has_questions',
                'employment_type', 'required_experience', 'required_education',
                'industry', 'function'
            ]
            
            # Fill missing columns with appropriate default values
            for col in expected_columns:
                if col not in df.columns:
                    if col in ['telecommuting', 'has_company_logo', 'has_questions']:
                        # Boolean columns
                        df[col] = False
                    elif col in ['job_id']:
                        # ID columns
                        df[col] = 'unknown'
                    else:
                        # Text columns
                        df[col] = ''
            
            # Convert boolean columns to proper format if they come as strings
            boolean_columns = ['telecommuting', 'has_company_logo', 'has_questions']
            for col in boolean_columns:
                if col in df.columns:
                    if isinstance(df[col].iloc[0], str):
                        df[col] = df[col].map({'true': True, 'false': False, 'True': True, 'False': False}).fillna(False)
                    df[col] = df[col].astype(bool)
            
            # Ensure text columns are strings and handle NaN values
            text_columns = ['title', 'location', 'department', 'salary_range',
                        'company_profile', 'description', 'requirements', 'benefits',
                        'employment_type', 'required_experience', 'required_education',
                        'industry', 'function']
            
            for col in text_columns:
                if col in df.columns:
                    # Convert to string and handle NaN/None values
                    df[col] = df[col].astype(str).fillna('').replace('nan', '').replace('None', '')
                    # Remove any remaining object dtypes that might cause issues
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else '')
            
            # Debug: Print data types before preprocessing
            logging.info(f"Data types before preprocessing: {df.dtypes.to_dict()}")
            logging.info(f"Sample data: {df.iloc[0].to_dict()}")
            
            # Use the preprocessor's transform_test_data method to process the data
            try:
                processed_features = self.preprocessor.transform_test_data(df)
            except Exception as e:
                logging.error(f"Error in preprocessor transform: {str(e)}")
                # If the preprocessor fails, try alternative approach
                processed_features = self.preprocessor.transform(df)
            
            # Ensure the output is in the right format
            if hasattr(processed_features, 'dtype'):
                logging.info(f"Processed features dtype: {processed_features.dtype}")
            
            # Convert to dense array if it's sparse and causing issues
            if hasattr(processed_features, 'todense'):
                processed_features = processed_features.todense()
            
            # Ensure numeric types
            if hasattr(processed_features, 'astype'):
                processed_features = processed_features.astype(np.float64)
            
            return processed_features
        
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            logging.error(f"Input data types: {df.dtypes.to_dict() if 'df' in locals() else 'DataFrame not created'}")
            raise
        
    def predict(self, job_data):
        """
        Predict if a job posting is fraudulent
        
        Args:
            job_data: Dictionary containing job posting information
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess the data using the trained preprocessor
            X = self.preprocess_job_posting(job_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]
            
            # Get confidence scores
            fraud_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            legitimate_probability = prediction_proba[0] if len(prediction_proba) > 1 else 1 - prediction_proba[0]
            
            return {
                'is_fraud': bool(prediction),
                'fraud_probability': float(fraud_probability),
                'legitimate_probability': float(legitimate_probability),
                'confidence': float(max(prediction_proba))
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
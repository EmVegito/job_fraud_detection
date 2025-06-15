from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback
from werkzeug.exceptions import BadRequest
from src.logger import logging
from src.api.endpoints import FraudJobDetector

app = Flask(__name__)

detector = FraudJobDetector(
    model_path='./data/models/final_model.pkl',  # Update with your model path
    preprocessor_path='./data/models/preprocessor.pkl'  # Update with your preprocessor path
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """
    Predict if a job posting is fraudulent
    
    Expected JSON payload:
    {
        "job_id": "12345",
        "title": "Data Scientist",
        "location": "New York, NY",
        "department": "Technology",
        "salary_range": "$80,000 - $120,000",
        "company_profile": "Leading tech company...",
        "description": "We are looking for an experienced data scientist...",
        "requirements": "Bachelor's degree in Computer Science...",
        "benefits": "Health insurance, 401k, flexible hours...",
        "telecommuting": false,
        "has_company_logo": true,
        "has_questions": true,
        "employment_type": "Full-time",
        "required_experience": "3-5 years",
        "required_education": "Bachelor's degree",
        "industry": "Technology",
        "function": "Data Science"
    }
    """
    try:
        # Validate content type
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Get job data from request
        job_data = request.get_json()
        
        if not job_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields (checking for both presence and non-empty values)
        required_fields = ['job_id', 'title', 'location', 'department', 
                        'salary_range', 'company_profile', 'description',
                        'requirements', 'benefits', 'telecommuting', 'has_company_logo', 'has_questions',
                        'employment_type', 'required_experience', 'required_education', 'industry', 'function']
        
        missing_fields = []
        for field in required_fields:
            if field not in job_data:
                missing_fields.append(field)
            elif job_data[field] is None:
                missing_fields.append(f"{field} (is None)")
            elif field in ['telecommuting', 'has_company_logo', 'has_questions']:
                # Boolean fields - just check for presence
                continue
            elif isinstance(job_data[field], str) and job_data[field].strip() == '':
                missing_fields.append(f"{field} (is empty)")
        
        if missing_fields:
            return jsonify({
                'error': f'Missing or invalid required fields: {missing_fields}'
            }), 400
        
        # Make prediction
        result = detector.predict(job_data)
        
        # Add metadata
        response = {
            'prediction': result,
            'job_title': job_data.get('title', ''),
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'  # Update with your model version
        }
        
        return jsonify(response), 200
        
    except BadRequest:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple job postings at once
    
    Expected JSON payload:
    {
        "jobs": [
            {job_posting_1},
            {job_posting_2},
            ...
        ]
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if 'jobs' not in data or not isinstance(data['jobs'], list):
            return jsonify({'error': 'Expected "jobs" array in request body'}), 400
        
        if len(data['jobs']) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size cannot exceed 100 jobs'}), 400
        
        results = []
        
        for i, job_data in enumerate(data['jobs']):
            try:
                prediction = detector.predict(job_data)
                results.append({
                    'index': i,
                    'job_title': job_data.get('title', ''),
                    'prediction': prediction,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'job_title': job_data.get('title', ''),
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_loaded': detector.model is not None,
        'preprocessor_loaded': detector.preprocessor is not None,
        'model_type': str(type(detector.model).__name__) if detector.model else None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    try:
        # Update these paths with your actual model files
        if not detector.model:
            logging.warning("Model not loaded. Please update model_path in FraudJobDetector initialization")
        
        if not detector.preprocessor:
            logging.warning("Preprocessor not loaded. Please update preprocessor_path in FraudJobDetector initialization")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False  # Set to False in production
        )
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")
        raise
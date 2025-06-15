import pandas as pd
import numpy as np
import scipy.sparse
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split

class JobFraudDataPipeline:
    """
    A comprehensive pipeline for job fraud detection data preprocessing and feature engineering.
    Enhanced with outlier handling and proper data leakage prevention.
    """
    
    def __init__(self, max_tfidf_features=500, test_size=0.2, random_state=42, 
                 outlier_method='iqr', outlier_threshold_multiplier=1.5):
        """
        Initialize the pipeline with configurable parameters.
        
        Parameters:
        max_tfidf_features (int): Maximum number of TF-IDF features per text column
        test_size (float): Proportion of data to use for validation
        random_state (int): Random state for reproducibility
        outlier_method (str): 'iqr' or 'zscore' for outlier detection
        outlier_threshold_multiplier (float): Multiplier for outlier thresholds
        """
        self.max_tfidf_features = max_tfidf_features
        self.test_size = test_size
        self.random_state = random_state
        self.outlier_method = outlier_method
        self.outlier_threshold_multiplier = outlier_threshold_multiplier
        
        # Store fitted transformers for potential future use
        self.fitted_tfidf_vectorizers = {}
        self.fitted_ohe_encoder = None
        self.fitted_scaler = None
        self.numerical_columns = None
        self.outlier_bounds = {}  # Store outlier bounds fitted on training data
        
        # Define column groups
        self.text_cols = ['company_profile', 'description', 'requirements', 'benefits']
        self.categorical_cols = ['location', 'employment_type', 'required_experience', 
                               'required_education', 'industry', 'function']
        self.text_columns_for_tfidf = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        self.categorical_cols_for_ohe = ['employment_type', 'required_experience', 'required_education',
                                       'industry', 'function', 'country']
        self.columns_for_log_transformation = ['company_profile_len', 'company_profile_word_count', 'description_len', 
                                             'description_word_count', 'requirements_len', 'requirements_word_count', 
                                             'benefits_len', 'benefits_word_count', 'title_len', 
                                             'title_word_count']
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        pd.DataFrame: Dataframe with missing values handled
        """
        print("--- Handling Missing Values ---")
        
        # Create binary features for high-missingness columns
        df['has_salary_range'] = df['salary_range'].notna().astype(int)
        df['has_department'] = df['department'].notna().astype(int)
        
        # Drop columns with too many missing values
        df = df.drop(['salary_range', 'department'], axis=1)
        
        # Handle text columns
        for col in self.text_cols:
            df[f'has_{col}'] = df[col].notna().astype(int)  # Binary feature for missingness
            df[col] = df[col].fillna('no_information_provided')  # Impute with placeholder
        
        # Handle categorical columns
        for col in self.categorical_cols:
            df[col] = df[col].fillna('Not Specified')
        
        print("Missing values handled successfully.")
        print(f"Remaining null values: {df.isnull().sum().sum()}")
        
        return df
    
    def engineer_features(self, df):
        """
        Perform feature engineering on the dataset.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        pd.DataFrame: Dataframe with engineered features
        """
        print("--- Feature Engineering ---")
        
        # Extract features from text columns
        for col in self.text_cols:
            df[f'{col}_len'] = df[col].apply(lambda x: len(str(x)))
            df[f'{col}_word_count'] = df[col].apply(lambda x: len(str(x).split()))
        
        # Extract features from 'title' column
        df['title_len'] = df['title'].apply(lambda x: len(str(x)))
        df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
        
        # Extract country from location
        df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip())
        
        print("Feature engineering completed.")
        
        return df
    
    def detect_outliers_bounds(self, df, columns):
        """
        Detect outlier bounds using the specified method (fitted on training data only).
        
        Parameters:
        df (pd.DataFrame): Input dataframe (training data)
        columns (list): List of columns to detect outliers for
        
        Returns:
        dict: Dictionary with outlier bounds for each column
        """
        print(f"--- Detecting Outlier Bounds using {self.outlier_method.upper()} method ---")
        
        bounds = {}
        
        for col in columns:
            if col in df.columns:
                data = df[col].values
                
                if self.outlier_method == 'iqr':
                    # Use IQR method
                    Q1 = np.percentile(data, 25)
                    Q3 = np.percentile(data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.outlier_threshold_multiplier * IQR
                    upper_bound = Q3 + self.outlier_threshold_multiplier * IQR
                    
                elif self.outlier_method == 'zscore':
                    # Use Z-score method
                    mean = np.mean(data)
                    std = np.std(data)
                    lower_bound = mean - self.outlier_threshold_multiplier * std
                    upper_bound = mean + self.outlier_threshold_multiplier * std
                
                bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
                
                # Count outliers for reporting
                outlier_count = np.sum((data < lower_bound) | (data > upper_bound))
                print(f"  - {col}: bounds [{lower_bound:.2f}, {upper_bound:.2f}], outliers: {outlier_count}")
        
        return bounds
    
    def handle_outliers(self, df, fit=True):
        """
        Handle outliers in numerical columns using capping (Winsorization).
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        fit (bool): Whether to fit outlier bounds (True for training, False for test)
        
        Returns:
        pd.DataFrame: Dataframe with outliers handled
        """
        print("--- Handling Outliers ---")
        
        # Define columns that might have outliers (numerical features from feature engineering)
        outlier_columns = [col for col in self.columns_for_log_transformation if col in df.columns]
        
        if fit:
            # Fit outlier bounds on training data
            self.outlier_bounds = self.detect_outliers_bounds(df, outlier_columns)
        
        if not self.outlier_bounds:
            print("No outlier bounds available. Skipping outlier handling.")
            return df
        
        df_processed = df.copy()
        total_capped = 0
        
        for col in outlier_columns:
            if col in self.outlier_bounds:
                bounds = self.outlier_bounds[col]
                
                # Count values that will be capped
                below_lower = np.sum(df_processed[col] < bounds['lower'])
                above_upper = np.sum(df_processed[col] > bounds['upper'])
                
                # Cap outliers (Winsorization)
                df_processed[col] = np.clip(df_processed[col], bounds['lower'], bounds['upper'])
                
                capped_count = below_lower + above_upper
                total_capped += capped_count
                
                if capped_count > 0:
                    print(f"  - {col}: capped {capped_count} outliers ({below_lower} below, {above_upper} above)")
        
        print(f"Total outliers capped: {total_capped}")
        
        return df_processed
    
    def apply_log_transformation(self, df):
        """
        Apply log transformation to specified columns to reduce skewness.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        pd.DataFrame: Dataframe with log-transformed features
        """
        print("--- Applying Log Transformation ---")
        
        df_processed = df.copy()
        
        for col in self.columns_for_log_transformation:
            if col in df_processed.columns:
                df_processed[f'{col}_log'] = np.log1p(df_processed[col])
                print(f"  - Applied log transformation to {col}")
        
        return df_processed
    
    def create_tfidf_features(self, df, fit=True):
        """
        Create TF-IDF features from text columns.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        fit (bool): Whether to fit vectorizers (True for training, False for test)
        
        Returns:
        scipy.sparse matrix: Combined TF-IDF features
        """
        print("--- TF-IDF Vectorization ---")
        
        tfidf_features = []
        
        for col in self.text_columns_for_tfidf:
            if fit:
                vectorizer = TfidfVectorizer(max_features=self.max_tfidf_features, stop_words='english')
                col_tfidf = vectorizer.fit_transform(df[col])
                self.fitted_tfidf_vectorizers[col] = vectorizer
            else:
                if col not in self.fitted_tfidf_vectorizers:
                    raise ValueError(f"Vectorizer for column '{col}' not fitted")
                vectorizer = self.fitted_tfidf_vectorizers[col]
                col_tfidf = vectorizer.transform(df[col])
            
            tfidf_features.append(col_tfidf)
            print(f"  - '{col}' TF-IDF shape: {col_tfidf.shape}")
        
        all_tfidf_features = scipy.sparse.hstack(tfidf_features)
        print(f"Combined TF-IDF features shape: {all_tfidf_features.shape}")
        
        return all_tfidf_features
    
    def create_categorical_features(self, df, fit=True):
        """
        Create one-hot encoded features from categorical columns.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        fit (bool): Whether to fit encoder (True for training, False for test)
        
        Returns:
        scipy.sparse matrix: One-hot encoded features
        """
        print("--- One-Hot Encoding ---")
        
        if fit:
            self.fitted_ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            encoded_features = self.fitted_ohe_encoder.fit_transform(df[self.categorical_cols_for_ohe])
        else:
            if self.fitted_ohe_encoder is None:
                raise ValueError("OneHotEncoder not fitted")
            encoded_features = self.fitted_ohe_encoder.transform(df[self.categorical_cols_for_ohe])
        
        print(f"  - One-Hot Encoded features shape: {encoded_features.shape}")
        
        return encoded_features
    
    def create_numerical_features(self, df):
        """
        Extract numerical and binary features.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        scipy.sparse matrix: Numerical and binary features
        """
        print("--- Extracting Numerical/Binary Features ---")
        
        # Define columns to exclude from numerical features
        columns_to_exclude = (self.text_columns_for_tfidf + 
                            self.categorical_cols_for_ohe + 
                            ['job_id', 'location', 'fraudulent'] +
                            self.columns_for_log_transformation)  # Exclude original columns if log versions exist
        
        # Include log-transformed versions if they exist
        log_columns = [f'{col}_log' for col in self.columns_for_log_transformation if f'{col}_log' in df.columns]
        
        columns_to_keep = [col for col in df.columns if col not in columns_to_exclude] + log_columns
        
        # Remove duplicates and ensure consistent ordering
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        
        # Store numerical columns for consistency in transform
        if self.numerical_columns is None:
            self.numerical_columns = columns_to_keep
        
        df_numerical_binary = df[self.numerical_columns].copy()
        numerical_binary_sparse = scipy.sparse.csr_matrix(df_numerical_binary.values)
        
        print(f"  - Numerical/Binary features shape: {numerical_binary_sparse.shape}")
        print(f"  - Numerical columns: {len(self.numerical_columns)}")
        
        return numerical_binary_sparse
    
    def combine_features(self, numerical_features, tfidf_features, categorical_features):
        """
        Combine all feature matrices into a single sparse matrix.
        
        Parameters:
        numerical_features: Numerical/binary features sparse matrix
        tfidf_features: TF-IDF features sparse matrix
        categorical_features: One-hot encoded features sparse matrix
        
        Returns:
        scipy.sparse matrix: Combined feature matrix
        """
        print("--- Combining All Features ---")
        
        X = scipy.sparse.hstack([numerical_features, tfidf_features, categorical_features])
        print(f"Final Features (X) shape: {X.shape}")
        
        return X
    
    def scale_features(self, X, fit=True):
        """
        Scale features using MaxAbsScaler (works well with sparse matrices).
        
        Parameters:
        X: Feature matrix (sparse or dense)
        fit (bool): Whether to fit the scaler (True for training, False for test)
        
        Returns:
        Scaled feature matrix
        """
        if fit:
            print("--- Fitting and Scaling Features ---")
            self.fitted_scaler = MaxAbsScaler()
            X_scaled = self.fitted_scaler.fit_transform(X)
            print(f"  - Features scaled using MaxAbsScaler")
        else:
            print("--- Scaling Features (using fitted scaler) ---")
            if self.fitted_scaler is None:
                raise ValueError("Scaler must be fitted before transforming")
            X_scaled = self.fitted_scaler.transform(X)
            print(f"  - Features scaled using fitted MaxAbsScaler")
        
        print(f"  - Scaled features shape: {X_scaled.shape}")
        return X_scaled
    
    def _process_data(self, df, fit=True):
        """
        Internal method to process data through all transformation steps.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        fit (bool): Whether to fit transformers
        
        Returns:
        scipy.sparse matrix: Processed feature matrix
        """
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # Step 3: Apply log transformation
        df_processed = self.apply_log_transformation(df_processed)

        # Step 4: Handle outliers
        df_processed = self.handle_outliers(df_processed, fit=fit)
        
        # Step 5: Create TF-IDF features
        tfidf_features = self.create_tfidf_features(df_processed, fit=fit)
        
        # Step 6: Create categorical features
        categorical_features = self.create_categorical_features(df_processed, fit=fit)
        
        # Step 7: Create numerical features
        numerical_features = self.create_numerical_features(df_processed)
        
        # Step 8: Combine all features
        X = self.combine_features(numerical_features, tfidf_features, categorical_features)
        
        # Step 9: Scale features
        X_scaled = self.scale_features(X, fit=fit)
        
        return X_scaled, df_processed
    
    def fit_transform(self, df):
        """
        Complete pipeline to transform raw dataframe into train/validation sets.
        PROPERLY PREVENTS DATA LEAKAGE by splitting first, then fitting only on training data.
        
        Parameters:
        df (pd.DataFrame): Raw input dataframe
        
        Returns:
        tuple: (X_train, X_valid, y_train, y_valid)
        """
        print("=== Starting Job Fraud Detection Data Pipeline ===\n")
        print("=== STEP 1: SPLITTING DATA FIRST TO PREVENT LEAKAGE ===\n")
        
        # CRITICAL: Split the data FIRST before any transformations
        
        df_train, df_valid = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state, 
            stratify=df['fraudulent']
        )
        
        print(f"Training set size: {len(df_train)}")
        print(f"Validation set size: {len(df_valid)}")
        print(f"Training fraud rate: {df_train['fraudulent'].mean():.3f}")
        print(f"Validation fraud rate: {df_valid['fraudulent'].mean():.3f}")
        
        print("\n=== STEP 2: FITTING TRANSFORMERS ON TRAINING DATA ONLY ===\n")
        
        # Fit all transformers on training data only
        X_train_scaled, df_train_processed = self._process_data(df_train, fit=True)
        y_train = df_train_processed['fraudulent']
        
        print("\n=== STEP 3: TRANSFORMING VALIDATION DATA ===\n")
        
        # Transform validation data using fitted transformers (no fitting!)
        X_valid_scaled, df_valid_processed = self._process_data(df_valid, fit=False)
        y_valid = df_valid_processed['fraudulent']
        
        print(f"\nFinal shapes:")
        print(f"X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
        print(f"X_valid: {X_valid_scaled.shape}, y_valid: {y_valid.shape}")
        
        print("\n=== Pipeline Completed Successfully - NO DATA LEAKAGE ===")
        
        return X_train_scaled, X_valid_scaled, y_train, y_valid
    
    def get_feature_names(self):
        """
        Generate feature names for the final feature matrix.
        This is useful for model interpretation.
        
        Returns:
        list: List of feature names
        """
        if not self.fitted_tfidf_vectorizers or not self.fitted_ohe_encoder or not self.fitted_scaler:
            raise ValueError("Pipeline must be fitted before getting feature names")
        
        feature_names = []
        
        # Add numerical/binary feature names
        if self.numerical_columns:
            feature_names.extend(self.numerical_columns)
        
        # Add TF-IDF feature names
        for col in self.text_columns_for_tfidf:
            tfidf_names = [f'{col}_tfidf_{word}' for word in 
                          self.fitted_tfidf_vectorizers[col].get_feature_names_out()]
            feature_names.extend(tfidf_names)
        
        # Add one-hot encoded feature names
        ohe_names = self.fitted_ohe_encoder.get_feature_names_out(self.categorical_cols_for_ohe)
        feature_names.extend(ohe_names)
        
        return feature_names
    
    def save_pipeline(self, filepath):
        """
        Save the fitted pipeline to a pickle file.
        
        Parameters:
        filepath (str): Path where to save the pipeline
        """
        if not self.fitted_tfidf_vectorizers or not self.fitted_ohe_encoder or not self.fitted_scaler:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'fitted_tfidf_vectorizers': self.fitted_tfidf_vectorizers,
            'fitted_ohe_encoder': self.fitted_ohe_encoder,
            'fitted_scaler': self.fitted_scaler,
            'outlier_bounds': self.outlier_bounds,
            'numerical_columns': self.numerical_columns,
            'max_tfidf_features': self.max_tfidf_features,
            'outlier_method': self.outlier_method,
            'outlier_threshold_multiplier': self.outlier_threshold_multiplier,
            'text_cols': self.text_cols,
            'categorical_cols': self.categorical_cols,
            'text_columns_for_tfidf': self.text_columns_for_tfidf,
            'categorical_cols_for_ohe': self.categorical_cols_for_ohe,
            'columns_for_log_transformation': self.columns_for_log_transformation
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to: {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath):
        """
        Load a fitted pipeline from a pickle file.
        
        Parameters:
        filepath (str): Path to the saved pipeline
        
        Returns:
        JobFraudDataPipeline: Loaded pipeline instance
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create new instance
        pipeline = cls(
            max_tfidf_features=pipeline_data['max_tfidf_features'],
            outlier_method=pipeline_data.get('outlier_method', 'iqr'),
            outlier_threshold_multiplier=pipeline_data.get('outlier_threshold_multiplier', 1.5)
        )
        
        # Restore fitted components
        pipeline.fitted_tfidf_vectorizers = pipeline_data['fitted_tfidf_vectorizers']
        pipeline.fitted_ohe_encoder = pipeline_data['fitted_ohe_encoder']
        pipeline.fitted_scaler = pipeline_data['fitted_scaler']
        pipeline.outlier_bounds = pipeline_data.get('outlier_bounds', {})
        pipeline.numerical_columns = pipeline_data['numerical_columns']
        pipeline.text_cols = pipeline_data['text_cols']
        pipeline.categorical_cols = pipeline_data['categorical_cols']
        pipeline.text_columns_for_tfidf = pipeline_data['text_columns_for_tfidf']
        pipeline.categorical_cols_for_ohe = pipeline_data['categorical_cols_for_ohe']
        pipeline.columns_for_log_transformation = pipeline_data.get('columns_for_log_transformation', [])
        
        print(f"Pipeline loaded from: {filepath}")
        return pipeline
    
    def transform_test_data(self, df_test):
        """
        Transform test data using the fitted pipeline (without target variable).
        
        Parameters:
        df_test (pd.DataFrame): Test dataframe (without 'fraudulent' column)
        
        Returns:
        scipy.sparse matrix: Transformed test features
        """
        if not self.fitted_tfidf_vectorizers or not self.fitted_ohe_encoder or not self.fitted_scaler:
            raise ValueError("Pipeline must be fitted before transforming test data")
        
        print("=== Transforming Test Data ===\n")
        
        # Transform test data using fitted transformers (no fitting!)
        X_test_scaled, _ = self._process_data(df_test, fit=False)
        
        print("=== Test Data Transformation Completed ===")
        return X_test_scaled

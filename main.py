import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/job_fraud_detection/training_data.csv')

# 1. Columns with a very high percentage of missing values (salary_range, department)
# Create new binary features indicating if the value is missing
df['has_salary_range'] = df['salary_range'].notna().astype(int)
df['has_department'] = df['department'].notna().astype(int)

# Drop the original columns as they have too many missing values to be directly useful without complex imputation
df = df.drop(['salary_range', 'department'], axis=1)

text_cols = ['company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[f'has_{col}'] = df[col].notna().astype(int) # Binary feature for missingness
    df[col] = df[col].fillna('no_information_provided') # Impute with placeholder

# 3. Categorical columns (location, employment_type, required_experience, required_education, industry, function)
categorical_cols = ['location', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
for col in categorical_cols:
    df[col] = df[col].fillna('Not Specified')

# Check null values after imputation
print("\nNull values after imputation:")
print(df.isnull().sum())

# Feature Engineering: Extract features from text columns
for col in text_cols:
    df[f'{col}_len'] = df[col].apply(lambda x: len(str(x))) # Length of the text
    df[f'{col}_word_count'] = df[col].apply(lambda x: len(str(x).split())) # Word count

# Extract features from 'title' column
df['title_len'] = df['title'].apply(lambda x: len(str(x)))
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

# Convert `location` to a more general form (e.g., Country)
# For simplicity, let's extract the first part of the location string (country or state)
df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip())


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical features for outlier detection (excluding binary and job_id)
numerical_features = [
    'company_profile_len', 'company_profile_word_count',
    'description_len', 'description_word_count',
    'requirements_len', 'requirements_word_count',
    'benefits_len', 'benefits_word_count',
    'title_len', 'title_word_count'
]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse


print("\n--- Performing Advanced Feature Engineering ---")

# Text-based columns for TF-IDF Vectorization
text_columns_for_tfidf = ['title', 'company_profile', 'description', 'requirements', 'benefits']

# Initialize TF-IDF Vectorizers and create a list to hold sparse matrices
tfidf_features = []
fitted_tfidf_vectorizers = {}

print("TF-IDF Vectorization for text columns:")
for col in text_columns_for_tfidf:
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english') # Max 500 features per text column
    col_tfidf = vectorizer.fit_transform(df[col])
    tfidf_features.append(col_tfidf)
    fitted_tfidf_vectorizers[col] = vectorizer # Store fitted vectorizer
    print(f"  - '{col}' TF-IDF shape: {col_tfidf.shape}")

all_tfidf_features = scipy.sparse.hstack(tfidf_features)
print(f"Combined TF-IDF features shape: {all_tfidf_features.shape}")

# Categorical columns for One-Hot Encoding
categorical_cols_for_ohe = [
    'employment_type', 'required_experience', 'required_education',
    'industry', 'function', 'country'
]

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
print("\nOne-Hot Encoding for categorical columns:")
encoded_features = encoder.fit_transform(df[categorical_cols_for_ohe])
print(f"  - One-Hot Encoded features shape: {encoded_features.shape}")
ohe_col_names = encoder.get_feature_names_out(categorical_cols_for_ohe)

# --- Define Features (X) and Target (y) - CRITICAL CORRECTION FOR DATA LEAKAGE ---
# Exclude the target variable 'fraudulent' and other non-feature columns
columns_to_keep_for_X = [col for col in df.columns if col not in text_columns_for_tfidf and \
                         col not in categorical_cols_for_ohe and \
                         col not in ['job_id', 'location', 'fraudulent']] # <-- 'fraudulent' is explicitly excluded here

# Extract numerical and binary features that are part of X
df_numerical_binary_features_X = df[columns_to_keep_for_X].copy()

# Convert these numerical/binary features to a sparse matrix
numerical_binary_sparse_X = scipy.sparse.csr_matrix(df_numerical_binary_features_X.values)

# Combine all features (X) into one final sparse matrix
X = scipy.sparse.hstack([numerical_binary_sparse_X, all_tfidf_features, encoded_features])

# Define the target variable (y)
y = df['fraudulent']

print(f"\nFinal Features (X) shape: {X.shape}")
print(f"Target Variable (y) shape: {y.shape}")
print("\nData is now correctly prepared for model training, with features (X) and target (y) separated.")

# --- Optional: Display a sample of the Features DataFrame (X) ---
# This conversion to a dense DataFrame can be memory-intensive for large datasets.
print("\n--- Sample of Features (X) DataFrame (first 5 rows, subset of columns) ---")
print("Note: Creating this dense sample is for display purposes and can be memory intensive for large datasets.")
try:
    # Generate column names for X only
    tfidf_feature_names_X = []
    for col in text_columns_for_tfidf:
        for i in range(fitted_tfidf_vectorizers[col].vocabulary_.__len__()):
            tfidf_feature_names_X.append(f'{col}_tfidf_{i}')

    final_X_column_names = list(df_numerical_binary_features_X.columns) + \
                         tfidf_feature_names_X + \
                         list(ohe_col_names)

    if X.shape[1] == len(final_X_column_names):
        sample_X_rows = X[:5].toarray()
        # Display first 20 columns to fit output, adjust as needed
        sample_X_df = pd.DataFrame(sample_X_rows[:, :20], columns=final_X_column_names[:20])
        print(sample_X_df)
    else:
        print("Column name mismatch for sample display. Skipping sample DataFrame creation.")
except MemoryError:
    print("Could not create sample X DataFrame due to memory constraints.")
except Exception as e:
    print(f"Error displaying sample X DataFrame: {e}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

import numpy as np
from scipy.sparse import csr_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

# Assuming you have X_train (CSR matrix) and y_train (binary labels)
# X_train should be your compressed sparse row matrix
# y_train should be your binary labels (0 and 1)

def apply_smote_to_sparse_data(X_train, y_train, random_state=42):
    """
    Apply SMOTE to handle class imbalance in sparse matrix data
    
    Parameters:
    X_train: scipy.sparse.csr_matrix - Training features
    y_train: array-like - Binary labels (0 and 1)
    random_state: int - Random state for reproducibility
    
    Returns:
    X_resampled: scipy.sparse.csr_matrix - Resampled features
    y_resampled: array - Resampled labels
    """
    
    # Check original class distribution
    original_distribution = Counter(y_train)
    print("Original class distribution:")
    print(f"Class 0: {original_distribution[0]} samples")
    print(f"Class 1: {original_distribution[1]} samples")
    print(f"Imbalance ratio: {original_distribution[0]/original_distribution[1]:.2f}")
    
    # Initialize SMOTE
    smote = SMOTE(random_state=random_state)
    
    # Apply SMOTE - it can handle sparse matrices directly
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Check new class distribution
    new_distribution = Counter(y_resampled)
    print("\nAfter SMOTE class distribution:")
    print(f"Class 0: {new_distribution[0]} samples")
    print(f"Class 1: {new_distribution[1]} samples")
    print(f"Imbalance ratio: {new_distribution[0]/new_distribution[1]:.2f}")
    
    return X_resampled, y_resampled

def plot_class_distribution(y_original, y_resampled):
    """
    Plot class distribution before and after SMOTE
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original distribution
    original_counts = Counter(y_original)
    ax1.bar(original_counts.keys(), original_counts.values(), color=['red', 'blue'], alpha=0.7)
    ax1.set_title('Original Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xticks([0, 1])
    
    # After SMOTE distribution
    resampled_counts = Counter(y_resampled)
    ax2.bar(resampled_counts.keys(), resampled_counts.values(), color=['red', 'blue'], alpha=0.7)
    ax2.set_title('After SMOTE Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xticks([0, 1])
    
    plt.tight_layout()
    plt.show()

# Apply SMOTE to your data
X_train_balanced, y_train_balanced = apply_smote_to_sparse_data(X_train, y_train)

# Optional: Plot the distributions
plot_class_distribution(y_train, y_train_balanced)

# Verify the result is still a sparse matrix
print(f"\nOriginal X_train shape: {X_train.shape}")
print(f"Original X_train type: {type(X_train)}")
print(f"Balanced X_train shape: {X_train_balanced.shape}")
print(f"Balanced X_train type: {type(X_train_balanced)}")

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_balanced, y_train_balanced)

y_pred = log_reg.predict(X_valid)

print(accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))
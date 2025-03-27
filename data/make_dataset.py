#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes the raw dataset into a format suitable for machine learning models.
It handles data cleaning, preprocessing, feature extraction, and splitting into train/test sets.
"""

import os
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(input_filepath):
    """
    Load the raw dataset from the specified file path.
    
    Args:
        input_filepath (str): Path to the raw dataset file.
        
    Returns:
        pandas.DataFrame: The loaded raw dataset.
    """
    logger.info(f'Loading dataset from {input_filepath}')
    
    # Determine file type and load accordingly
    file_ext = os.path.splitext(input_filepath)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_filepath)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(input_filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    logger.info(f'Loaded dataset with shape: {df.shape}')
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicates, and inconsistent data.
    
    Args:
        df (pandas.DataFrame): Raw dataset.
        
    Returns:
        pandas.DataFrame: Cleaned dataset.
    """
    logger.info('Cleaning dataset...')
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Record initial shape
    initial_shape = df_clean.shape
    logger.info(f'Initial dataset shape: {initial_shape}')
    
    # Check for and remove duplicates
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        logger.info(f'Removing {duplicate_count} duplicate rows')
        df_clean = df_clean.drop_duplicates()
    
    # Check for missing values
    missing_values = df_clean.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index.tolist()
    
    if missing_columns:
        logger.info(f'Columns with missing values: {missing_columns}')
        
        # Identify categorical and numerical columns with missing values
        cat_columns = df_clean.select_dtypes(include=['object', 'category']).columns
        num_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
        
        cat_missing = [col for col in missing_columns if col in cat_columns]
        num_missing = [col for col in missing_columns if col in num_columns]
        
        # Impute missing values
        if cat_missing:
            logger.info(f'Imputing missing categorical values using mode for columns: {cat_missing}')
            for col in cat_missing:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
        
        if num_missing:
            logger.info(f'Imputing missing numerical values using median for columns: {num_missing}')
            for col in num_missing:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
    
    # Check for inconsistent categorical values (standardize case, strip whitespace)
    for col in df_clean.select_dtypes(include=['object']):
        if df_clean[col].dtype == 'object':
            # Strip whitespace
            df_clean[col] = df_clean[col].str.strip()
            
            # Standardize case (to title case for readability)
            df_clean[col] = df_clean[col].str.title()
    
    # Handle outliers in numerical columns using IQR method
    for col in df_clean.select_dtypes(include=['int64', 'float64']):
        # Skip identifier columns and target
        if col == 'Target' or col.lower().endswith('id'):
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)][col]
        if len(outliers) > 0:
            logger.info(f'Capping {len(outliers)} outliers in column: {col}')
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Log final shape
    final_shape = df_clean.shape
    logger.info(f'Final cleaned dataset shape: {final_shape}')
    logger.info(f'Removed rows: {initial_shape[0] - final_shape[0]}')
    
    return df_clean


def extract_features(df):
    """
    Extract and create features from the cleaned dataset.
    
    Args:
        df (pandas.DataFrame): Cleaned dataset.
        
    Returns:
        pandas.DataFrame: Dataset with extracted features.
    """
    logger.info('Extracting and creating features...')
    
    # Make a copy to avoid modifying the original
    df_featured = df.copy()
    
    # Create academic performance indicators
    logger.info('Creating academic performance indicators...')
    
    # 1. Success Rate (Approved vs. Enrolled)
    df_featured['1st_sem_success_rate'] = df_featured['Curricular units 1st sem (approved)'] / df_featured['Curricular units 1st sem (enrolled)'].replace(0, 1)
    df_featured['2nd_sem_success_rate'] = df_featured['Curricular units 2nd sem (approved)'] / df_featured['Curricular units 2nd sem (enrolled)'].replace(0, 1)
    df_featured['overall_success_rate'] = (df_featured['Curricular units 1st sem (approved)'] + df_featured['Curricular units 2nd sem (approved)']) / \
                               (df_featured['Curricular units 1st sem (enrolled)'] + df_featured['Curricular units 2nd sem (enrolled)']).replace(0, 1)
    
    # Replace infinity with 0
    df_featured.replace([np.inf, -np.inf], 0, inplace=True)
    
    # 2. Evaluation Engagement Ratio
    df_featured['1st_sem_evaluation_ratio'] = df_featured['Curricular units 1st sem (evaluations)'] / df_featured['Curricular units 1st sem (enrolled)'].replace(0, 1)
    df_featured['2nd_sem_evaluation_ratio'] = df_featured['Curricular units 2nd sem (evaluations)'] / df_featured['Curricular units 2nd sem (enrolled)'].replace(0, 1)
    
    # 3. Performance Trend (2nd semester vs 1st semester)
    df_featured['grade_trend'] = df_featured['Curricular units 2nd sem (grade)'] - df_featured['Curricular units 1st sem (grade)']
    df_featured['approval_trend'] = df_featured['Curricular units 2nd sem (approved)'] - df_featured['Curricular units 1st sem (approved)']
    
    # 4. Weighted Grade (considering number of units)
    df_featured['weighted_grade'] = (df_featured['Curricular units 1st sem (grade)'] * df_featured['Curricular units 1st sem (enrolled)'] + 
                               df_featured['Curricular units 2nd sem (grade)'] * df_featured['Curricular units 2nd sem (enrolled)']) / \
                              (df_featured['Curricular units 1st sem (enrolled)'] + df_featured['Curricular units 2nd sem (enrolled)']).replace(0, 1)
    
    # 5. Non-evaluation Rate
    df_featured['1st_sem_non_eval_rate'] = df_featured['Curricular units 1st sem (without evaluations)'] / df_featured['Curricular units 1st sem (enrolled)'].replace(0, 1)
    df_featured['2nd_sem_non_eval_rate'] = df_featured['Curricular units 2nd sem (without evaluations)'] / df_featured['Curricular units 2nd sem (enrolled)'].replace(0, 1)
    
    # 6. Credit Efficiency
    df_featured['credit_efficiency'] = (df_featured['Curricular units 1st sem (credited)'] + df_featured['Curricular units 2nd sem (credited)']) / \
                                 (df_featured['Curricular units 1st sem (enrolled)'] + df_featured['Curricular units 2nd sem (enrolled)']).replace(0, 1)
    
    # Create engagement metrics
    logger.info('Creating engagement metrics...')
    
    # 1. Overall Engagement Score
    df_featured['engagement_score'] = ((df_featured['Curricular units 1st sem (evaluations)'] / df_featured['Curricular units 1st sem (enrolled)'].replace(0, 1)) + 
                                (df_featured['Curricular units 2nd sem (evaluations)'] / df_featured['Curricular units 2nd sem (enrolled)'].replace(0, 1))) / 2 * 100
    
    # 2. Dropout Risk Indicator
    df_featured['dropout_risk_indicator'] = (df_featured['1st_sem_non_eval_rate'] + df_featured['2nd_sem_non_eval_rate']) / 2 * 100
    
    # 3. Academic Consistency
    df_featured['academic_consistency'] = np.where(
        (df_featured['1st_sem_success_rate'] > 0) & (df_featured['2nd_sem_success_rate'] > 0),
        100 - (abs(df_featured['1st_sem_success_rate'] - df_featured['2nd_sem_success_rate']) * 50),
        0  # If either semester has zero success rate, consistency is 0
    )
    
    # 4. Academic Momentum
    df_featured['academic_momentum'] = np.where(
        df_featured['approval_trend'] > 0,
        df_featured['approval_trend'] * 10,  # Positive momentum
        df_featured['approval_trend'] * 5    # Negative momentum (less weight)
    )
    
    # Create socioeconomic indicators
    logger.info('Creating socioeconomic indicators...')
    
    # 1. Financial Status Indicator (combining scholarship, debtor status, tuition fees)
    df_featured['financial_status'] = df_featured['Scholarship holder'] * 5 + (1 - df_featured['Debtor']) * 5 + (df_featured['Tuition fees up to date']) * 5
    
    # 2. Financial Risk (debtor status and tuition payment)
    df_featured['financial_risk'] = df_featured['Debtor'] * 5 + (1 - df_featured['Tuition fees up to date']) * 5
    
    # Create economic context features
    logger.info('Creating economic context features...')
    
    # 1. Economic Pressure Index (combination of unemployment and inflation)
    df_featured['economic_pressure'] = df_featured['Unemployment rate'] + df_featured['Inflation rate'] - df_featured['GDP'] / 100
    
    # 2. Economic Risk for Non-Scholarship Students
    df_featured['economic_risk_non_scholarship'] = np.where(
        df_featured['Scholarship holder'] == 0,
        df_featured['economic_pressure'] * 1.5,  # Higher risk for non-scholarship students
        df_featured['economic_pressure'] * 0.5   # Lower risk for scholarship students
    )
    
    # Log the number of created features
    original_cols = set(df.columns)
    new_cols = set(df_featured.columns) - original_cols
    logger.info(f'Created {len(new_cols)} new features: {sorted(new_cols)}')
    
    return df_featured


def encode_categorical_features(df, categorical_features):
    """
    Encode categorical features using one-hot encoding and label encoding.
    
    Args:
        df (pandas.DataFrame): Dataset with features.
        categorical_features (list): List of categorical feature names.
        
    Returns:
        tuple: (Transformed DataFrame, encoders dictionary)
    """
    logger.info('Encoding categorical features...')
    
    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Initialize encoders dictionary to store for later use
    encoders = {}
    
    # Identify features for one-hot encoding (low cardinality) and label encoding (high cardinality)
    one_hot_features = []
    label_features = []
    
    for feature in categorical_features:
        if feature not in df_encoded.columns:
            logger.warning(f"Feature '{feature}' not found in the dataframe. Skipping.")
            continue
            
        # Check cardinality
        unique_count = df_encoded[feature].nunique()
        
        if unique_count <= 10:  # Threshold for one-hot encoding
            one_hot_features.append(feature)
        else:
            label_features.append(feature)
    
    logger.info(f'Features for one-hot encoding ({len(one_hot_features)}): {one_hot_features}')
    logger.info(f'Features for label encoding ({len(label_features)}): {label_features}')
    
    # Perform one-hot encoding
    if one_hot_features:
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        
        # Fit and transform
        encoded_array = ohe.fit_transform(df_encoded[one_hot_features])
        
        # Get feature names
        feature_names = []
        for i, feature in enumerate(one_hot_features):
            categories = ohe.categories_[i][1:]  # Drop first category
            feature_names.extend([f"{feature}_{cat}" for cat in categories])
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_encoded.index)
        
        # Combine with original DataFrame
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        # Store encoder
        encoders['one_hot'] = {
            'encoder': ohe,
            'features': one_hot_features,
            'encoded_features': feature_names
        }
    
    # Perform label encoding
    if label_features:
        label_encoders = {}
        
        for feature in label_features:
            le = LabelEncoder()
            df_encoded[f"{feature}_encoded"] = le.fit_transform(df_encoded[feature])
            label_encoders[feature] = le
        
        # Store encoders
        encoders['label'] = {
            'encoders': label_encoders,
            'features': label_features,
            'encoded_features': [f"{feature}_encoded" for feature in label_features]
        }
    
    # Encode target if it's categorical
    if 'Target' in df_encoded.columns and df_encoded['Target'].dtype == 'object':
        target_encoder = LabelEncoder()
        df_encoded['Target_encoded'] = target_encoder.fit_transform(df_encoded['Target'])
        
        # Store mapping for interpretation
        target_mapping = dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))
        logger.info(f'Target mapping: {target_mapping}')
        
        # Store encoder
        encoders['target'] = {
            'encoder': target_encoder,
            'mapping': target_mapping
        }
    
    return df_encoded, encoders


def scale_numerical_features(df, numerical_features, method='standard'):
    """
    Scale numerical features using the specified method.
    
    Args:
        df (pandas.DataFrame): Dataset with features.
        numerical_features (list): List of numerical feature names.
        method (str): Scaling method ('standard', 'minmax', or 'robust').
        
    Returns:
        tuple: (Transformed DataFrame, scaler)
    """
    logger.info(f'Scaling numerical features using {method} scaling...')
    
    # Make a copy to avoid modifying the original
    df_scaled = df.copy()
    
    # Filter out features not in the dataframe
    features_to_scale = [f for f in numerical_features if f in df_scaled.columns]
    
    if not features_to_scale:
        logger.warning('No features to scale.')
        return df_scaled, None
    
    # Create scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit and transform
    scaled_array = scaler.fit_transform(df_scaled[features_to_scale])
    
    # Create DataFrame with scaled features
    scaled_feature_names = [f"{col}_scaled" for col in features_to_scale]
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_feature_names, index=df_scaled.index)
    
    # Combine with original DataFrame
    df_scaled = pd.concat([df_scaled, scaled_df], axis=1)
    
    logger.info(f'Scaled {len(features_to_scale)} features')
    
    return df_scaled, scaler


def split_data(df, target_col='Target_encoded', test_size=0.2, val_size=0.1):
    """
    Split data into training, validation, and test sets.
    
    Args:
        df (pandas.DataFrame): Dataset with features and target.
        target_col (str): Target column name.
        test_size (float): Proportion of data for test set.
        val_size (float): Proportion of data for validation set.
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    logger.info('Splitting data into train, validation, and test sets...')
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataframe.")
    
    # Calculate the effective validation size relative to the remaining data after test split
    effective_val_size = val_size / (1 - test_size)
    
    # First split: training + validation vs. test
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df[target_col]
    )
    
    # Second split: training vs. validation
    train, val = train_test_split(
        train_val, 
        test_size=effective_val_size, 
        random_state=42, 
        stratify=train_val[target_col]
    )
    
    logger.info(f'Train set: {train.shape[0]} samples ({train.shape[0]/df.shape[0]:.1%})')
    logger.info(f'Validation set: {val.shape[0]} samples ({val.shape[0]/df.shape[0]:.1%})')
    logger.info(f'Test set: {test.shape[0]} samples ({test.shape[0]/df.shape[0]:.1%})')
    
    return train, val, test


def save_processed_data(train_data, val_data, test_data, encoders, scaler, output_dir):
    """
    Save processed data and preprocessing objects to disk.
    
    Args:
        train_data (pandas.DataFrame): Training dataset.
        val_data (pandas.DataFrame): Validation dataset.
        test_data (pandas.DataFrame): Test dataset.
        encoders (dict): Dictionary of encoders.
        scaler (object): Fitted scaler.
        output_dir (str): Output directory path.
    """
    logger.info(f'Saving processed data to {output_dir}')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save encoders and scaler
    with open(os.path.join(output_dir, 'feature_info.json'), 'w') as f:
        import json
        json.dump(feature_info, f, indent=4)
    
    logger.info('Processed data saved successfully')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--test-size', type=float, default=0.2, help='Proportion of data for test set')
@click.option('--val-size', type=float, default=0.1, help='Proportion of data for validation set')
@click.option('--scaling', type=click.Choice(['standard', 'minmax', 'robust']), default='standard', 
              help='Method for scaling numerical features')
def main(input_filepath, output_filepath, test_size, val_size, scaling):
    """
    Process raw data into cleaned and transformed data ready for modeling.
    
    Args:
        input_filepath (str): Path to the raw data file
        output_filepath (str): Directory to save processed data
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        scaling (str): Method for scaling numerical features
    """
    logger.info('Starting data processing pipeline...')
    
    # Load data
    df = load_dataset(input_filepath)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Identify categorical and numerical features
    categorical_features = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target if it's in the features lists
    if 'Target' in categorical_features:
        categorical_features.remove('Target')
    if 'Target' in numerical_features:
        numerical_features.remove('Target')
    
    logger.info(f'Categorical features: {len(categorical_features)}')
    logger.info(f'Numerical features: {len(numerical_features)}')
    
    # Extract features
    df_featured = extract_features(df_cleaned)
    
    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df_featured, categorical_features)
    
    # Store original feature lists
    encoders['categorical_features'] = categorical_features
    encoders['numerical_features'] = numerical_features
    
    # Update numerical features list with new features
    new_numerical_features = [col for col in df_encoded.columns 
                             if col not in categorical_features 
                             and col != 'Target'
                             and col != 'Target_encoded'
                             and df_encoded[col].dtype in ['int64', 'float64']]
    
    # Scale numerical features
    df_scaled, scaler = scale_numerical_features(df_encoded, new_numerical_features, method=scaling)
    
    # Split data
    train_data, val_data, test_data = split_data(df_scaled, test_size=test_size, val_size=val_size)
    
    # Save processed data
    save_processed_data(train_data, val_data, test_data, encoders, scaler, output_filepath)
    
    logger.info('Data processing completed successfully!')


if __name__ == '__main__':
    # Find .env file (if it exists)
    load_dotenv(find_dotenv())
    
    # Run main function
    main()
    with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
        
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature information
    feature_info = {
        'original_features': {
            'categorical': encoders.get('categorical_features', []),
            'numerical': encoders.get('numerical_features', [])
        },
        'encoded_features': {
            'one_hot': encoders.get('one_hot', {}).get('encoded_features', []),
            'label': encoders.get('label', {}).get('encoded_features', [])
        },
        'scaled_features': [col for col in train_data.columns if col.endswith('_scaled')],
        'target': encoders.get('target', {}).get('mapping', {})
    }
    
    with open(os.path.join(output_dir, 'feature_info.json'), 'w') as f:
        import json
        json.dump(feature_info, f, indent=4)
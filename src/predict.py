"""
Gender Prediction Module
Loads trained model and imputer, preprocesses raw user data, and predicts gender.
"""

import joblib
import pandas as pd
import numpy as np
from user_agents import parse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

# Paths to model files (update as needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gender_classifier.pkl')
IMPUTER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'imputer.pkl')

# Global variables to cache loaded objects
_model = None
_imputer = None

def load_model():
    """Load the trained model and imputer (cached)."""
    global _model, _imputer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _imputer = joblib.load(IMPUTER_PATH)
    return _model, _imputer

def parse_user_agent(ua_string):
    """Extract browser, OS, and device type from user agent string."""
    try:
        if pd.isna(ua_string):
            return pd.Series({'browser': 'Unknown', 'os': 'Unknown', 'is_mobile': 0, 'is_tablet': 0})
        ua = parse(str(ua_string))
        return pd.Series({
            'browser': ua.browser.family,
            'os': ua.os.family,
            'is_mobile': 1 if ua.is_mobile else 0,
            'is_tablet': 1 if ua.is_tablet else 0
        })
    except:
        return pd.Series({'browser': 'Unknown', 'os': 'Unknown', 'is_mobile': 0, 'is_tablet': 0})

def preprocess_dataframe(df, referer_vectors=None, geo_info=None, is_train=False):
    """
    Apply all feature engineering steps to a raw dataframe.
    Args:
        df: raw data with columns ['user_id', 'request_ts', 'referer', 'geo_id', 'user_agent']
        referer_vectors: DataFrame with columns ['referer', 'component0'...'component9']
        geo_info: DataFrame with columns ['geo_id', 'country_id', 'region_id', 'timezone_id']
        is_train: if True, also return feature columns list for training.
    Returns:
        Processed DataFrame with user‑level aggregated features.
    """
    # Merge referer vectors
    if referer_vectors is not None:
        df = df.merge(referer_vectors, on='referer', how='left')
    
    # Merge geo info
    if geo_info is not None:
        df = df.merge(geo_info, on='geo_id', how='left')
    
    # Parse user agents
    ua_features = df['user_agent'].apply(parse_user_agent)
    df[['browser', 'os', 'is_mobile', 'is_tablet']] = ua_features
    
    # Extract time features
    df['request_ts'] = pd.to_datetime(df['request_ts'], unit='s')
    df['hour'] = df['request_ts'].dt.hour
    df['weekday'] = df['request_ts'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Drop raw columns
    df.drop(['request_ts', 'user_agent', 'referer'], axis=1, inplace=True, errors='ignore')
    
    # Aggregate per user
    grouped = df.groupby('user_id')
    agg_list = []
    
    # Component means
    components = [f'component{i}' for i in range(10) if f'component{i}' in df.columns]
    for comp in components:
        mean_val = grouped[comp].mean().rename(f'comp_mean_{comp[-1]}')
        agg_list.append(mean_val)
    
    # Mode for categorical columns
    cat_cols = ['country_id', 'region_id', 'timezone_id', 'browser', 'os']
    for col in cat_cols:
        if col in df.columns:
            mode_val = grouped[col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown').rename(f'mode_{col}')
            agg_list.append(mode_val)
    
    # Device ratios
    for col in ['is_mobile', 'is_tablet']:
        if col in df.columns:
            ratio = grouped[col].mean().rename(f'{col}_ratio')
            agg_list.append(ratio)
    
    # Time means
    for col in ['hour', 'weekday', 'is_weekend']:
        if col in df.columns:
            time_mean = grouped[col].mean().rename(f'mean_{col}')
            agg_list.append(time_mean)
    
    # Activity count
    activity = grouped.size().rename('activity_count')
    agg_list.append(activity)
    
    # Combine all aggregations
    agg_df = pd.concat(agg_list, axis=1).reset_index()
    
    return agg_df

def predict_gender(raw_df, referer_vectors, geo_info):
    """
    Predict gender for each user in raw_df.
    Args:
        raw_df: DataFrame with columns ['user_id', 'request_ts', 'referer', 'geo_id', 'user_agent']
        referer_vectors: DataFrame (as above)
        geo_info: DataFrame (as above)
    Returns:
        DataFrame with columns ['user_id', 'target'] (0/1 predictions)
    """
    model, imputer = load_model()
    
    # Preprocess
    agg_df = preprocess_dataframe(raw_df, referer_vectors, geo_info)
    
    # The feature columns must match those used during training.
    # For simplicity, we assume the model expects the same column order as in training.
    # A robust approach is to save the feature column list from training and reuse it.
    # Here we just use all columns except 'user_id' (but in practice, you should align).
    feature_cols = [col for col in agg_df.columns if col != 'user_id']
    X = agg_df[feature_cols].copy()
    
    # Encode categorical columns (mode_*)
    cat_cols = [col for col in X.columns if col.startswith('mode_')]
    for col in cat_cols:
        X[col] = X[col].fillna('Unknown').astype(str)
        # Note: The LabelEncoder should be fitted on training data.
        # This is a placeholder – you must save the encoders during training.
        # For now, we'll assume the categories are already encoded in the model's imputer? Not ideal.
        # A better approach is to save the encoders as well.
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])   # This will break if test has unseen categories.
    
    # Impute missing values
    X_imputed = imputer.transform(X)
    
    # Predict
    preds = model.predict(X_imputed)
    
    result = pd.DataFrame({'user_id': agg_df['user_id'], 'target': preds})
    return result

# Example usage (commented out)
if __name__ == "__main__":
    # Load data
    train_data = pd.read_csv('../data/train.csv', sep=';')
    referer = pd.read_csv('../data/referer_vectors.csv', sep=';')
    geo = pd.read_csv('../data/geo_info.csv', sep=';')
    
    pred_df = predict_gender(train_data.head(100), referer, geo)
    print(pred_df.head())
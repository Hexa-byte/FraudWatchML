import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from scipy.spatial.distance import mahalanobis
from config import FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

# Don't import TensorFlow at module level to avoid startup issues
# These imports will be done lazily in the methods that need them

logger = logging.getLogger(__name__)

class DataPipeline:
    """Data pipeline for processing transaction data for fraud detection"""
    
    def __init__(self, db_uri=None):
        """Initialize the data pipeline
        
        Args:
            db_uri (str): Database URI for connecting to transaction data
        """
        self.db_uri = db_uri or os.environ.get("DATABASE_URL")
        self.engine = create_engine(self.db_uri) if self.db_uri else None
        self.schema = None
        
    def load_data_from_db(self, days=30, limit=None):
        """Load transaction data from database
        
        Args:
            days (int): Number of days of data to load
            limit (int): Maximum number of records to load
        
        Returns:
            pd.DataFrame: Loaded transaction data
        """
        if not self.engine:
            raise ValueError("Database connection not configured")
        
        logger.info(f"Loading transaction data from past {days} days")
        start_date = datetime.utcnow() - timedelta(days=days)
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = text(f"""
            SELECT * FROM transaction 
            WHERE timestamp >= '{start_date}'
            ORDER BY timestamp DESC
            {limit_clause}
        """)
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} transactions")
            return df
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
    
    def preprocess_data(self, df, training=False):
        """Preprocess transaction data for model input
        
        Args:
            df (pd.DataFrame): Raw transaction data
            training (bool): Whether preprocessing is for training
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Preprocessing transaction data")
        
        # Handle missing values
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # Feature engineering
        if 'timestamp' in df.columns:
            # Extract time-based features
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Handle categorical features
        df = self._encode_categorical_features(df)
        
        # Detect outliers
        if training:
            df = self._handle_outliers(df)
        
        logger.info("Preprocessing complete")
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical features
        
        Args:
            df (pd.DataFrame): DataFrame with categorical features
        
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                # One-hot encode categorical columns
                one_hot = pd.get_dummies(df[col], prefix=col)
                # Drop the original column
                df = df.drop(col, axis=1)
                # Join the encoded columns
                df = df.join(one_hot)
        
        return df
    
    def _handle_outliers(self, df):
        """Detect and handle outliers using Mahalanobis distance
        
        Args:
            df (pd.DataFrame): Transaction data
        
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        numeric_df = df[NUMERIC_COLUMNS].copy()
        
        # Calculate mean and covariance
        mean = numeric_df.mean()
        cov = numeric_df.cov()
        inv_cov = np.linalg.inv(cov)
        
        # Calculate Mahalanobis distance
        distances = []
        for i, row in numeric_df.iterrows():
            distances.append(mahalanobis(row, mean, inv_cov))
        
        df['mahalanobis_distance'] = distances
        
        # Mark outliers (distance > 3 std deviations)
        threshold = np.mean(distances) + 3 * np.std(distances)
        df['is_outlier'] = df['mahalanobis_distance'] > threshold
        
        logger.info(f"Detected {df['is_outlier'].sum()} outliers out of {len(df)} records")
        
        return df
    
    def generate_tf_schema(self, df):
        """Generate TensorFlow Data Validation schema
        
        Args:
            df (pd.DataFrame): Sample transaction data
        
        Returns:
            Schema: TensorFlow data schema
        """
        # Lazy import TensorFlow dependencies
        try:
            import tensorflow_data_validation as tfdv
            from tensorflow_transform.tf_metadata import metadata_io
        except ImportError:
            logger.error("TensorFlow Data Validation not available")
            return None
            
        logger.info("Generating TensorFlow data schema")
        
        stats = tfdv.generate_statistics_from_dataframe(df)
        schema = tfdv.infer_schema(stats)
        
        # Save schema for future use
        schema_file = os.path.join(os.getcwd(), 'data', 'schema.pbtxt')
        os.makedirs(os.path.dirname(schema_file), exist_ok=True)
        metadata_io.write_metadata_text(schema, schema_file)
        
        self.schema = schema
        logger.info(f"Schema generated and saved to {schema_file}")
        
        return schema
    
    def validate_data(self, df):
        """Validate transaction data against schema
        
        Args:
            df (pd.DataFrame): Transaction data to validate
        
        Returns:
            bool: Whether data is valid
        """
        # Lazy import TensorFlow dependencies
        try:
            import tensorflow_data_validation as tfdv
        except ImportError:
            logger.error("TensorFlow Data Validation not available")
            return False
            
        if self.schema is None:
            logger.warning("Schema not available, generating from current data")
            self.generate_tf_schema(df)
            
        if self.schema is None:
            logger.error("Could not generate schema for validation")
            return False
        
        logger.info("Validating data against schema")
        stats = tfdv.generate_statistics_from_dataframe(df)
        anomalies = tfdv.validate_statistics(stats, schema=self.schema)
        
        if anomalies.anomaly_info:
            logger.warning(f"Data validation found anomalies: {anomalies.anomaly_info}")
            return False
        
        logger.info("Data validation successful")
        return True
    
    def create_tf_dataset(self, df, batch_size=32, shuffle=True):
        """Create TensorFlow dataset from DataFrame
        
        Args:
            df (pd.DataFrame): Preprocessed transaction data
            batch_size (int): Batch size for dataset
            shuffle (bool): Whether to shuffle the dataset
        
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Lazy import TensorFlow
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not available")
            return None
            
        # Extract features and labels if available
        if 'is_fraud' in df.columns:
            features = df.drop(['is_fraud', 'id', 'timestamp'], errors='ignore')
            labels = df['is_fraud'].values
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        else:
            features = df.drop(['id', 'timestamp'], errors='ignore')
            dataset = tf.data.Dataset.from_tensor_slices(dict(features))
        
        # Prepare dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def handle_class_imbalance(self, df):
        """Handle class imbalance in training data
        
        Args:
            df (pd.DataFrame): Transaction data with labels
        
        Returns:
            pd.DataFrame: Balanced dataset
        """
        if 'is_fraud' not in df.columns:
            logger.warning("No fraud labels in data, skipping class imbalance handling")
            return df
        
        fraud_count = df['is_fraud'].sum()
        non_fraud_count = len(df) - fraud_count
        
        logger.info(f"Class distribution - Fraud: {fraud_count}, Non-fraud: {non_fraud_count}")
        
        # If already balanced or no fraud cases, return as is
        if fraud_count / len(df) > 0.3 or fraud_count == 0:
            return df
        
        # Oversample fraud cases
        fraud_df = df[df['is_fraud'] == 1]
        non_fraud_df = df[df['is_fraud'] == 0]
        
        # Calculate how many fraud samples we need
        desired_fraud_ratio = 0.3
        desired_fraud_count = int(non_fraud_count * desired_fraud_ratio / (1 - desired_fraud_ratio))
        
        # If we need more fraud samples than we have, oversample with replacement
        if desired_fraud_count > fraud_count:
            fraud_df_oversampled = fraud_df.sample(n=desired_fraud_count, replace=True)
            balanced_df = pd.concat([non_fraud_df, fraud_df_oversampled])
            logger.info(f"Oversampled fraud cases from {fraud_count} to {desired_fraud_count}")
        else:
            # Otherwise, just use a subset of the non-fraud cases
            non_fraud_df_undersampled = non_fraud_df.sample(n=int(fraud_count / desired_fraud_ratio) - fraud_count)
            balanced_df = pd.concat([non_fraud_df_undersampled, fraud_df])
            logger.info(f"Undersampled non-fraud cases from {non_fraud_count} to {len(non_fraud_df_undersampled)}")
        
        return balanced_df.sample(frac=1).reset_index(drop=True)  # Shuffle

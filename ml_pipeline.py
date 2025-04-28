import os
import argparse
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from data_pipeline import DataPipeline
from model_architecture import AutoencoderModel, GradientBoostedTreesModel
from utils import log_audit
from models import ModelVersion, db
from app import app
from config import (
    TRAINING_EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED,
    FEATURE_COLUMNS, AUTOENCODER_LATENT_DIM
)

logger = logging.getLogger(__name__)

def train_models(data_path=None, epochs=None, batch_size=None):
    """Train fraud detection models
    
    Args:
        data_path (str): Path to transaction data CSV
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (autoencoder_path, gbm_path, precision, recall)
    """
    # Use default values if not provided
    epochs = epochs or TRAINING_EPOCHS
    batch_size = batch_size or BATCH_SIZE
    
    logger.info(f"Starting model training with {epochs} epochs and batch size {batch_size}")
    
    # Initialize data pipeline
    data_pipeline = DataPipeline()
    
    # Load data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Loading data from database")
        df = data_pipeline.load_data_from_db(days=90)
    
    if df is None or len(df) == 0:
        logger.error("No data available for training")
        return None, None, 0, 0
    
    logger.info(f"Loaded {len(df)} transactions for training")
    
    # Preprocess data
    df = data_pipeline.preprocess_data(df, training=True)
    
    # Handle class imbalance
    if 'is_fraud' in df.columns:
        df = data_pipeline.handle_class_imbalance(df)
    
    # Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    
    logger.info(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
    
    # Train autoencoder model
    logger.info("Training autoencoder model")
    
    # Prepare data for autoencoder (use only non-fraud transactions)
    if 'is_fraud' in train_df.columns:
        autoencoder_train_df = train_df[train_df['is_fraud'] == 0].drop(['is_fraud'], axis=1)
        autoencoder_val_df = val_df[val_df['is_fraud'] == 0].drop(['is_fraud'], axis=1)
    else:
        autoencoder_train_df = train_df.copy()
        autoencoder_val_df = val_df.copy()
    
    # Remove non-feature columns
    for df in [autoencoder_train_df, autoencoder_val_df]:
        for col in ['id', 'transaction_id', 'timestamp']:
            if col in df.columns:
                df.drop([col], axis=1, inplace=True)
    
    # Create TensorFlow datasets
    train_dataset = data_pipeline.create_tf_dataset(autoencoder_train_df, batch_size=batch_size)
    val_dataset = data_pipeline.create_tf_dataset(autoencoder_val_df, batch_size=batch_size)
    
    # Initialize and train autoencoder
    autoencoder = AutoencoderModel(input_dim=len(autoencoder_train_df.columns), latent_dim=AUTOENCODER_LATENT_DIM)
    autoencoder_history = autoencoder.train(train_dataset, val_dataset, epochs=epochs, batch_size=batch_size)
    
    # Save autoencoder model
    autoencoder_path = autoencoder.save_model()
    
    # Train GBM model
    logger.info("Training Gradient Boosted Trees model")
    
    # Prepare data for GBM (needs labeled data)
    if 'is_fraud' in train_df.columns:
        # GBM needs features and labels
        gbm_train_features = train_df.drop(['is_fraud', 'id', 'transaction_id', 'timestamp'], errors='ignore')
        gbm_train_labels = train_df['is_fraud'].values
        
        gbm_val_features = val_df.drop(['is_fraud', 'id', 'transaction_id', 'timestamp'], errors='ignore')
        gbm_val_labels = val_df['is_fraud'].values
        
        # Create TensorFlow datasets
        gbm_train_dataset = tf.data.Dataset.from_tensor_slices((dict(gbm_train_features), gbm_train_labels))
        gbm_train_dataset = gbm_train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        gbm_val_dataset = tf.data.Dataset.from_tensor_slices((dict(gbm_val_features), gbm_val_labels))
        gbm_val_dataset = gbm_val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Initialize and train GBM
        gbm = GradientBoostedTreesModel()
        gbm_history = gbm.train(gbm_train_dataset, gbm_val_dataset)
        
        # Save GBM model
        gbm_path = gbm.save_model()
        
        # Calculate metrics
        precision, recall = evaluate_model(autoencoder, gbm, val_df)
    else:
        logger.warning("No fraud labels in data, skipping GBM training")
        gbm_path = None
        precision, recall = 0, 0
    
    return autoencoder_path, gbm_path, precision, recall

def evaluate_model(autoencoder, gbm, val_df):
    """Evaluate model performance
    
    Args:
        autoencoder: Trained autoencoder model
        gbm: Trained GBM model
        val_df: Validation DataFrame
        
    Returns:
        tuple: (precision, recall)
    """
    if 'is_fraud' not in val_df.columns:
        logger.warning("No fraud labels in validation data, cannot evaluate model")
        return 0, 0
    
    # Prepare data
    features = val_df.drop(['is_fraud', 'id', 'transaction_id', 'timestamp'], errors='ignore')
    labels = val_df['is_fraud'].values
    
    # Compute autoencoder reconstruction error
    reconstruction_error = autoencoder.compute_reconstruction_error(features)
    
    # Normalize reconstruction error to 0-1 range
    min_error, max_error = np.min(reconstruction_error), np.max(reconstruction_error)
    normalized_error = (reconstruction_error - min_error) / (max_error - min_error)
    
    # Get GBM predictions
    gbm_data = tf.data.Dataset.from_tensor_slices(dict(features)).batch(32)
    gbm_preds = gbm.model.predict(gbm_data)
    
    # Combine scores
    combined_score = 0.7 * normalized_error + 0.3 * gbm_preds
    
    # Apply threshold
    predictions = combined_score > 0.5
    
    # Calculate metrics
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    logger.info(f"Model evaluation - Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return precision, recall

def register_model(autoencoder_path, gbm_path, precision, recall):
    """Register a trained model in the database
    
    Args:
        autoencoder_path (str): Path to saved autoencoder model
        gbm_path (str): Path to saved GBM model
        precision (float): Model precision
        recall (float): Model recall
        
    Returns:
        ModelVersion: Registered model version
    """
    with app.app_context():
        try:
            # Generate version string
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            version = f"v{timestamp}"
            
            # Create model version record
            model_version = ModelVersion(
                version=version,
                created_at=datetime.utcnow(),
                active=False,  # Don't automatically activate
                autoencoder_path=autoencoder_path,
                gbm_path=gbm_path,
                precision=precision,
                recall=recall
            )
            
            db.session.add(model_version)
            db.session.commit()
            
            logger.info(f"Model registered with version {version}")
            
            log_audit(
                action="register_model",
                resource_type="model_version",
                resource_id=model_version.id,
                details=f"Registered model version {version} with precision {precision:.4f} and recall {recall:.4f}"
            )
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            db.session.rollback()
            return None

def activate_model(model_id):
    """Activate a model version
    
    Args:
        model_id (int): ID of the model version to activate
        
    Returns:
        bool: Success status
    """
    with app.app_context():
        try:
            # Deactivate current active model
            ModelVersion.query.filter_by(active=True).update({'active': False})
            
            # Activate new model
            model = ModelVersion.query.get(model_id)
            if not model:
                logger.error(f"Model with ID {model_id} not found")
                return False
            
            model.active = True
            db.session.commit()
            
            logger.info(f"Activated model version {model.version}")
            
            log_audit(
                action="activate_model",
                resource_type="model_version",
                resource_id=model_id,
                details=f"Activated model version {model.version}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating model: {e}")
            db.session.rollback()
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register fraud detection models")
    parser.add_argument("--data_path", type=str, help="Path to transaction data CSV")
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--activate", action="store_true", help="Activate the model after training")
    
    args = parser.parse_args()
    
    # Train models
    autoencoder_path, gbm_path, precision, recall = train_models(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if autoencoder_path and gbm_path:
        # Register model
        model_version = register_model(autoencoder_path, gbm_path, precision, recall)
        
        if model_version and args.activate:
            # Activate model
            activate_model(model_version.id)
            logger.info(f"Model version {model_version.version} activated")
    else:
        logger.error("Model training failed")

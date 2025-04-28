import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import logging
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logger = logging.getLogger(__name__)

class AutoencoderModel:
    """Autoencoder model for unsupervised anomaly detection"""
    
    def __init__(self, input_dim, latent_dim=16, model_path=None):
        """Initialize the autoencoder model
        
        Args:
            input_dim (int): Input dimension (number of features)
            latent_dim (int): Dimension of the latent space
            model_path (str): Path to load a saved model
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the autoencoder architecture with 64-32-16-32-64 neurons"""
        # Define encoder
        inputs = Input(shape=(self.input_dim,), name='encoder_input')
        
        # Encoder layers (64-32-16)
        encoded = Dense(64, activation='relu', name='encoder_dense_1')(inputs)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(32, activation='relu', name='encoder_dense_2')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(self.latent_dim, activation='relu', name='encoder_dense_3')(encoded)
        
        # Define decoder
        # Decoder layers (16-32-64-input_dim)
        decoded = Dense(32, activation='relu', name='decoder_dense_1')(encoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(64, activation='relu', name='decoder_dense_2')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid', name='decoder_output')(decoded)
        
        # Create autoencoder model
        self.model = Model(inputs, decoded, name='autoencoder')
        
        # Create encoder model
        self.encoder = Model(inputs, encoded, name='encoder')
        
        # Create decoder model
        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer1 = self.model.get_layer('decoder_dense_1')
        decoder_layer2 = self.model.get_layer('decoder_dense_2')
        decoder_layer3 = self.model.get_layer('decoder_output')
        decoder_bn1 = self.model.get_layer('batch_normalization_2')
        decoder_bn2 = self.model.get_layer('batch_normalization_3')
        
        decoder_output = decoder_layer1(encoded_input)
        decoder_output = decoder_bn1(decoder_output)
        decoder_output = decoder_layer2(decoder_output)
        decoder_output = decoder_bn2(decoder_output)
        decoder_output = decoder_layer3(decoder_output)
        
        self.decoder = Model(encoded_input, decoder_output, name='decoder')
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae'])
        
        logger.info("Autoencoder model built with architecture: 64-32-16-32-64")
    
    def train(self, train_dataset, validation_dataset=None, epochs=100, batch_size=32):
        """Train the autoencoder model
        
        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: TensorFlow dataset for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History: Training history
        """
        logger.info(f"Training autoencoder model for {epochs} epochs with batch size {batch_size}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
            ModelCheckpoint(filepath='models/autoencoder_checkpoint.h5', 
                            monitor='val_loss', 
                            save_best_only=True)
        ]
        
        # Create folder for model checkpoints
        os.makedirs('models', exist_ok=True)
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        
        logger.info(f"Autoencoder training completed. Final loss: {history.history['loss'][-1]:.4f}")
        return history
    
    def save_model(self, model_dir='models'):
        """Save the autoencoder model
        
        Args:
            model_dir (str): Directory to save the model
        
        Returns:
            str: Path to the saved model
        """
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'autoencoder_{timestamp}.h5')
        
        self.model.save(model_path)
        logger.info(f"Autoencoder model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved autoencoder model
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            self.model = load_model(model_path)
            logger.info(f"Autoencoder model loaded from {model_path}")
            
            # Extract encoder and decoder parts
            encoder_layers = []
            decoder_layers = []
            found_latent = False
            
            for layer in self.model.layers:
                if 'encoder' in layer.name and not found_latent:
                    encoder_layers.append(layer)
                    if 'encoder_dense_3' in layer.name:
                        found_latent = True
                elif found_latent and 'decoder' in layer.name:
                    decoder_layers.append(layer)
            
            # Recreate encoder model
            encoder_input = self.model.input
            encoder_output = encoder_layers[-1].output
            self.encoder = Model(encoder_input, encoder_output, name='encoder')
            
            # Recreate decoder model
            decoder_input = Input(shape=(self.latent_dim,))
            decoder_output = decoder_input
            for layer in decoder_layers:
                decoder_output = layer(decoder_output)
            
            self.decoder = Model(decoder_input, decoder_output, name='decoder')
            
        except Exception as e:
            logger.error(f"Error loading autoencoder model: {e}")
            raise
    
    def compute_reconstruction_error(self, data):
        """Compute reconstruction error for anomaly detection
        
        Args:
            data: Input data
            
        Returns:
            numpy.ndarray: Reconstruction error for each sample
        """
        # Get predictions
        predictions = self.model.predict(data)
        
        # Compute MSE between input and output
        mse = tf.reduce_mean(tf.square(data - predictions), axis=1)
        
        return mse.numpy()


class GradientBoostedTreesModel:
    """Gradient Boosted Trees model for supervised fraud detection"""
    
    def __init__(self, model_path=None):
        """Initialize the GBM model
        
        Args:
            model_path (str): Path to load a saved model
        """
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the Gradient Boosted Trees model"""
        self.model = tfdf.keras.GradientBoostedTreesModel(
            # Model hyperparameters
            num_trees=200,
            max_depth=8,
            min_examples=20,
            learning_rate=0.1,
            # Training hyperparameters
            verbose=2
        )
        
        logger.info("Gradient Boosted Trees model created")
    
    def train(self, train_dataset, validation_dataset=None):
        """Train the GBM model
        
        Args:
            train_dataset: TensorFlow dataset for training
            validation_dataset: TensorFlow dataset for validation
            
        Returns:
            History: Training history
        """
        logger.info("Training Gradient Boosted Trees model")
        
        # Train the model
        history = self.model.fit(train_dataset, validation_data=validation_dataset)
        
        # Log variable importance
        logger.info("Feature importance:")
        for i, (name, importance) in enumerate(self.model.make_inspector().variable_importances()["MEAN_DECREASE_IN_ACCURACY"].items()):
            logger.info(f"  {i+1}. {name}: {importance:.4f}")
        
        logger.info("Gradient Boosted Trees model training completed")
        return history
    
    def save_model(self, model_dir='models'):
        """Save the GBM model
        
        Args:
            model_dir (str): Directory to save the model
        
        Returns:
            str: Path to the saved model
        """
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'gbm_{timestamp}')
        
        self.model.save(model_path)
        logger.info(f"GBM model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved GBM model
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            self.model = tfdf.keras.GradientBoostedTreesModel.load(model_path)
            logger.info(f"GBM model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading GBM model: {e}")
            raise
    
    def predict(self, data):
        """Make predictions with the GBM model
        
        Args:
            data: Input data
            
        Returns:
            numpy.ndarray: Fraud probability for each sample
        """
        return self.model.predict(data)


class HybridFraudModel:
    """Hybrid model combining autoencoder and GBM for fraud detection"""
    
    def __init__(self, autoencoder=None, gbm=None, autoencoder_weight=0.7):
        """Initialize the hybrid model
        
        Args:
            autoencoder: Autoencoder model
            gbm: Gradient Boosted Trees model
            autoencoder_weight (float): Weight for autoencoder scores (0-1)
        """
        self.autoencoder = autoencoder
        self.gbm = gbm
        self.autoencoder_weight = autoencoder_weight
        self.gbm_weight = 1.0 - autoencoder_weight
        
        logger.info(f"Hybrid model initialized with autoencoder weight {autoencoder_weight}")
    
    def predict_fraud(self, data, threshold=0.5):
        """Predict fraud using the hybrid model
        
        Args:
            data: Input data
            threshold (float): Fraud threshold (0-1)
            
        Returns:
            tuple: (fraud_scores, is_fraud)
        """
        # Get reconstruction error from autoencoder
        reconstruction_error = self.autoencoder.compute_reconstruction_error(data)
        
        # Normalize reconstruction error to 0-1 range
        # Use simple min-max scaling based on training set statistics
        # In a real system, you'd use a more sophisticated approach
        min_error, max_error = 0.01, 1.0  # These should be derived from training data
        normalized_error = (reconstruction_error - min_error) / (max_error - min_error)
        normalized_error = tf.clip_by_value(normalized_error, 0.0, 1.0)
        
        # Get fraud probability from GBM
        gbm_scores = self.gbm.predict(data)
        
        # Combine scores
        fraud_scores = (self.autoencoder_weight * normalized_error + 
                        self.gbm_weight * gbm_scores)
        
        # Apply threshold
        is_fraud = fraud_scores > threshold
        
        return fraud_scores, is_fraud
    
    def explain_prediction(self, data, feature_names):
        """Explain the fraud prediction
        
        Args:
            data: Input data (single sample)
            feature_names: List of feature names
            
        Returns:
            dict: Explanation of the prediction
        """
        # Get GBM feature importance for this prediction
        # This is a simplified version - in a real system, you'd use SHAP values
        variable_importances = self.gbm.model.make_inspector().variable_importances()
        
        # Get the top 5 most important features
        importance_dict = variable_importances["MEAN_DECREASE_IN_ACCURACY"]
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create explanation
        explanation = {
            "key_factors": [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in top_features
            ]
        }
        
        return explanation

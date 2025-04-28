import os
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
import tfx
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator
from tfx.components import Transform, Trainer, Evaluator, Pusher
from tfx.proto import example_gen_pb2
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_tfx_pipeline(
    pipeline_name, 
    pipeline_root, 
    data_path,
    serving_model_dir,
    metadata_path=None
):
    """Create TFX pipeline for fraud detection
    
    Args:
        pipeline_name (str): Name of the pipeline
        pipeline_root (str): Root directory for pipeline artifacts
        data_path (str): Path to training data
        serving_model_dir (str): Directory to export model for serving
        metadata_path (str): Path to store pipeline metadata
        
    Returns:
        tfx.dsl.Pipeline: TFX pipeline
    """
    logger.info(f"Creating TFX pipeline: {pipeline_name}")
    
    # Define pipeline input
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
        ])
    )
    
    example_gen = CsvExampleGen(
        input_base=data_path,
        output_config=output
    )
    
    # Generate statistics
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )
    
    # Generate schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )
    
    # Validate examples
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # Define preprocessing function
    def preprocessing_fn(inputs):
        """Preprocess input features for training
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of transformed features
        """
        # Define outputs
        outputs = {}
        
        # Normalize numeric features
        for feature in ['amount', 'latitude', 'longitude']:
            if feature in inputs:
                outputs[feature] = tft.scale_to_0_1(inputs[feature])
        
        # Pass through other numeric features
        for feature in ['hour_of_day', 'day_of_week', 'weekend']:
            if feature in inputs:
                outputs[feature] = inputs[feature]
        
        # One-hot encode categorical features
        for feature in ['payment_method', 'card_present']:
            if feature in inputs:
                outputs[f'{feature}_xf'] = tft.compute_and_apply_vocabulary(
                    inputs[feature], vocab_filename=feature)
        
        # Hash features with large cardinality
        for feature in ['device_id', 'ip_address']:
            if feature in inputs:
                outputs[f'{feature}_xf'] = tft.hash_strings(inputs[feature], 
                                                          num_buckets=100)
        
        # Pass through label
        if 'is_fraud' in inputs:
            outputs['is_fraud'] = inputs['is_fraud']
        
        return outputs
    
    # Transform features
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=preprocessing_fn
    )
    
    # Define trainer module
    def _create_trainer_module():
        """Create trainer module file for TFX
        
        Returns:
            str: Path to trainer module
        """
        trainer_module_path = os.path.join(pipeline_root, 'trainer_module.py')
        
        with open(trainer_module_path, 'w') as f:
            f.write('''
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_transform.tf_metadata import schema_utils

def _get_transformed_feature_spec(schema):
    """Get feature spec from schema
    
    Args:
        schema: TensorFlow metadata schema
        
    Returns:
        Feature spec for TF Transform
    """
    return schema_utils.schema_as_feature_spec(schema).feature_spec

def _build_autoencoder_model(input_shape):
    """Build autoencoder model
    
    Args:
        input_shape: Shape of input tensor
        
    Returns:
        tf.keras.Model: Autoencoder model
    """
    # Define encoder
    inputs = Input(shape=(input_shape,), name='encoder_input')
    
    # Encoder layers (64-32-16)
    encoded = Dense(64, activation='relu', name='encoder_dense_1')(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(32, activation='relu', name='encoder_dense_2')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(16, activation='relu', name='encoder_dense_3')(encoded)
    
    # Define decoder
    # Decoder layers (16-32-64-input_shape)
    decoded = Dense(32, activation='relu', name='decoder_dense_1')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(64, activation='relu', name='decoder_dense_2')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_shape, activation='sigmoid', name='decoder_output')(decoded)
    
    # Create model
    model = Model(inputs, decoded, name='autoencoder')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def _build_gbm_model():
    """Build gradient boosted trees model
    
    Returns:
        tf.keras.Model: GBM model
    """
    import tensorflow_decision_forests as tfdf
    
    # Create the model
    gbm = tfdf.keras.GradientBoostedTreesModel(
        # Model hyperparameters
        num_trees=200,
        max_depth=8,
        min_examples=20,
        learning_rate=0.1,
        # Training hyperparameters
        verbose=2
    )
    
    return gbm

def _input_fn(file_pattern, data_accessor, schema, batch_size=32):
    """Create input function for training
    
    Args:
        file_pattern: File pattern for input files
        data_accessor: TFX DataAccessor
        schema: TensorFlow metadata schema
        batch_size: Batch size for training
        
    Returns:
        tf.data.Dataset: Dataset for training/evaluation
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        schema=schema,
        batch_size=batch_size
    )
    
    def _extract_features_and_label(record):
        """Extract features and label from record
        
        Args:
            record: TensorFlow record
            
        Returns:
            (features, label): Tuple of features and label
        """
        features = record.copy()
        label = None
        
        if 'is_fraud' in features:
            label = features.pop('is_fraud')
        
        return features, label
    
    return dataset.map(_extract_features_and_label)

def run_fn(fn_args):
    """Run function for the trainer component
    
    Args:
        fn_args: Trainer function arguments
    """
    schema = tft.TFTransformOutput(fn_args.transform_output).transformed_metadata.schema
    
    # Create train and eval datasets
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=32
    )
    
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=32
    )
    
    # Get input shape
    for features_batch, _ in train_dataset.take(1):
        input_shape = sum([tf.shape(features_batch[key])[-1] for key in features_batch.keys()])
    
    # Build and train autoencoder model
    autoencoder = _build_autoencoder_model(input_shape)
    
    # Train the model
    autoencoder.fit(
        train_dataset,
        epochs=100,
        validation_data=eval_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Save autoencoder model
    autoencoder_path = os.path.join(fn_args.serving_model_dir, 'autoencoder')
    autoencoder.save(autoencoder_path)
    
    # Build and train GBM model
    gbm = _build_gbm_model()
    
    # Train the model
    gbm.fit(train_dataset, validation_data=eval_dataset)
    
    # Save GBM model
    gbm_path = os.path.join(fn_args.serving_model_dir, 'gbm')
    gbm.save(gbm_path)
    
    # Save the hybrid model with signature
    # In a real system, you'd create a hybrid model wrapper
    # For now, we just save both models separately
            ''')
        
        return trainer_module_path
    
    # Create training module
    trainer_module = _create_trainer_module()
    
    # Create trainer component
    trainer = Trainer(
        module_file=trainer_module,
        transformed_examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=10000),
        eval_args=tfx.proto.EvalArgs(num_steps=5000)
    )
    
    # Define evaluator
    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')
    
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(
                    signature_name='serving_default',
                    label_key='is_fraud'
                )
            ],
            slicing_specs=[
                tfma.SlicingSpec(),
                tfma.SlicingSpec(feature_keys=['payment_method'])
            ],
            metrics_specs=[
                tfma.MetricsSpec(
                    metrics=[
                        tfma.MetricConfig(class_name='AUC'),
                        tfma.MetricConfig(class_name='Precision'),
                        tfma.MetricConfig(class_name='Recall'),
                        tfma.MetricConfig(class_name='MeanSquaredError')
                    ]
                )
            ]
        )
    )
    
    # Define pusher
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )
    
    # Define pipeline components
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]
    
    # Create pipeline
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path)
    )
    
    return p

def run_pipeline():
    """Run the TFX pipeline"""
    # Set up paths
    pipeline_name = 'fraudwatch_pipeline'
    pipeline_root = os.path.join('tfx_pipeline', pipeline_name)
    data_path = os.path.join('data', 'transactions')
    serving_model_dir = os.path.join('models', 'serving')
    metadata_path = os.path.join('tfx_pipeline', 'metadata.sqlite')
    
    # Create directories
    os.makedirs(pipeline_root, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(serving_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Create and run pipeline
    tfx_pipeline = create_tfx_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_path=data_path,
        serving_model_dir=serving_model_dir,
        metadata_path=metadata_path
    )
    
    # Run the pipeline using Beam
    BeamDagRunner().run(tfx_pipeline)

if __name__ == "__main__":
    run_pipeline()

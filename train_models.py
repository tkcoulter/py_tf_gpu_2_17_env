import os
from load_data import load_and_prepare_dataset, stratified_train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, Masking
import numpy as np

def print_gpu_info():
    print("\nGPU Information:")
    print("-" * 50)
    
    # Check if TensorFlow can see any GPUs
    physical_devices = tf.config.list_physical_devices()
    print("All physical devices:", physical_devices)
    
    physical_gpus = tf.config.list_physical_devices('GPU')
    print("\nNumber of GPUs available:", len(physical_gpus))
    
    if len(physical_gpus) > 0:
        print("\nGPU Devices:")
        for gpu in physical_gpus:
            print(f"  {gpu}")
            
        # Try to get GPU device details
        try:
            for gpu in physical_gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"\nGPU Details: {details}")
        except:
            print("Could not get detailed GPU information")
    
    # Check if TensorFlow is using GPU
    print("\nTensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    print("TensorFlow GPU available:", tf.test.is_gpu_available())
    
    # Print device placement
    print("\nDevice Placement:")
    with tf.device('/CPU:0'):
        print("CPU operation:", tf.random.normal([1]).device)
    try:
        with tf.device('/GPU:0'):
            print("GPU operation:", tf.random.normal([1]).device)
    except:
        print("GPU operation: Not available")

def create_lstm_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    
    # Add masking layer to handle padding (-1 values)
    masked = Masking(mask_value=-1.0)(inputs)

    # Reduced model complexity
    x = Bidirectional(LSTM(64, return_sequences=True))(masked)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(16))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(output_shape, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=1000, batch_size=64):
    # Print device being used for training
    print("\nTraining Device Information:")
    print("-" * 50)
    tf.debugging.set_log_device_placement(True)  # This will show which device operations are running on
    
    # Convert lists to numpy arrays and get shapes
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, features)
    output_shape = y_train.shape[1]  # number of output features
    
    # Create and compile model
    model = create_lstm_model(input_shape, output_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    # Print model summary
    print("\nModel Summary:")
    print("-" * 50)
    model.summary()
    
    # Train model with smaller batch size
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

if __name__ == '__main__':
    print("\nStarting Training Script")
    print("=" * 50)
    
    # Print GPU information before starting
    print_gpu_info()
    
    print("\nWorking Directory:", os.getcwd())

    dataset_dir = "../scania/"
    print("Dataset Directory Contents:", sorted(os.listdir(dataset_dir)))

    # Clear memory before loading data
    tf.keras.backend.clear_session()
    
    df_train = load_and_prepare_dataset(dataset_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_train_test_split(df_train)

    y_train = [y[:,0] for y in y_train]
    y_val = [y[:,0] for y in y_val]
    y_test = [y[:,0] for y in y_test]

    # Train LSTM model
    model, history = train_lstm_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\nEvaluation Results:")
    print("-" * 50)
    test_loss, test_mae = model.evaluate(np.array(X_test), np.array(y_test), verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

import os
from load_data import load_and_prepare_dataset, stratified_train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, Masking
import numpy as np

def create_lstm_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    
    # Add masking layer to handle padding (-1 values)
    masked = Masking(mask_value=-1.0)(inputs)

    x = Bidirectional(LSTM(128, return_sequences=True))(masked)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = Bidirectional(LSTM(16))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(output_shape, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=1000, batch_size=256):
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
        optimizer='adam',
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
    
    # Train model
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
    print()
    print(os.getcwd())

    dataset_dir = "../scania/"
    print(sorted(os.listdir(dataset_dir)))

    df_train = load_and_prepare_dataset(dataset_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_train_test_split(df_train)

    y_train = [y[:,0] for y in y_train]
    y_val = [y[:,0] for y in y_val]
    y_val = [y[:,0] for y in y_test]

    # Train LSTM model
    model, history = train_lstm_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(np.array(X_test), np.array(y_test), verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def create_model(hidden_units=[16, 8], learning_rate=0.01):
    model = Sequential([
        Dense(hidden_units[0], input_shape=(7,), activation='sigmoid'),
        Dense(hidden_units[1], activation='sigmoid'),
        Dense(1, activation='sigmoid')  # Binary output
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

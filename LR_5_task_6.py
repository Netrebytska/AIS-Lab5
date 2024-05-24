#16 варіант

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def generate_data():
    x = np.linspace(-10, 10, 1000)
    y = 5 * x**2 + 7
    return x, y

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, x, y):
    model.fit(x, y, epochs=100, batch_size=32, verbose=0)
    return model

def plot_results(x, y, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Actual Data')
    plt.scatter(x, y_pred, label='Predicted Data')
    plt.legend()
    plt.show()

def main():
    x, y = generate_data()
    model = build_model()
    model = train_model(model, x, y)
    y_pred = model.predict(x)
    plot_results(x, y, y_pred)

if __name__ == "__main__":
    main()

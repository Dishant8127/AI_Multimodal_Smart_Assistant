# Simple LSTM text generation skeleton using TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(vocab_size, embedding_dim=128, rnn_units=256):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(rnn_units, return_sequences=True),
        layers.LSTM(rnn_units),
        layers.Dense(vocab_size, activation='softmax')
    ])
    return model

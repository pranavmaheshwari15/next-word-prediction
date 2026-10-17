from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2


def build_model(input_size, vocab_size):

    model = Sequential([
        Input(shape=(input_size,)),

        Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),

        Dropout(0.3),

        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),

        Dropout(0.2),

        Dense(vocab_size, activation="softmax")
    ])

    return model
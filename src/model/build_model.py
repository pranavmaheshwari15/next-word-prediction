from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def build_model(input_size, vocab_size):
    model = Sequential([
        Dense(128,activation = "relu",input_shape = (input_size,)),
        Dense(64,activation = "relu"),
        Dense(vocab_size,activation = "softmax")
    ])
    return model
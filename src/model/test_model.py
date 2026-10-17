from build_model import build_model
from compile_model import compile_model

input_size = 100
vocab_size = 1000

model = build_model(input_size, vocab_size)
model = compile_model(model)
model.summary()

print ("\nArchitecture test successful")
print("Input shape: ", model.input_shape)
print("Output shape: ", model.output_shape)
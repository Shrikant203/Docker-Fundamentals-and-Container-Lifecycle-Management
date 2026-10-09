import numpy as np

# Tiny 2-layer neural network forward pass (stand-in for a DL inference service)
rng = np.random.default_rng(42)
x = rng.normal(size=(4, 8))                  # batch of 4 samples, 8 features
W1, b1 = rng.normal(size=(8, 16)), np.zeros(16)
W2, b2 = rng.normal(size=(16, 3)), np.zeros(3)

h = np.maximum(0, x @ W1 + b1)               # ReLU hidden layer
logits = h @ W2 + b2
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

print("Containerized DL inference app")
print("NumPy version :", np.__version__)
print("Output shape  :", probs.shape)
print("Predicted classes:", probs.argmax(axis=1).tolist())

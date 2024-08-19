import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Create the model
model = Sequential()

# Add a hidden layer with 2 neurons and sigmoid activation
model.add(Dense(2, input_dim=2, activation='sigmoid'))

# Add the output layer with 1 neuron and sigmoid activation
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X, y, epochs=10000, verbose=0)
# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Make predictions
predictions = model.predict(X)
print("Predictions:")
print(np.round(predictions).astype(int))
    
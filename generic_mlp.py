import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_mlp(input_dim, output_dim, hidden_layers=[64, 32, 32], dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='SGD',
                  loss='crossentropy',
                  metrics=['accuracy'])

    return model

# Example data loading function (using Iris dataset)
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

# Preprocess data: encoding and standardization
def preprocess_data(X, y):
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_encoded


# Example usage
X, y = load_data()
X_scaled, y_encoded = preprocess_data(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create the model
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = create_mlp(input_dim=input_dim, output_dim=output_dim)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")


# Predict on new data
new_data = X_test  # Example: take the first 5 samples from the test set
predictions = model.predict(new_data)

# Convert predictions from one-hot encoded format back to class labels
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted classes: {predicted_classes}")

# Convert true labels from one-hot encoded format back to class labels
true_classes = np.argmax(y_test, axis=1)
print(f"True classes: {true_classes}")